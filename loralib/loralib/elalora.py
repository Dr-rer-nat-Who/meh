import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer 
from typing import Optional, List
from .utils import plot_rank, plot_ipt_graph
import os

import json
import numpy as np

class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                try:
                    result += (
                        self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                    ) * self.scaling / (self.ranknum+1e-5)
                except Exception as e:
                    print(e)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500,
        k: int = 2,
        b: int = 4,
        output_dir: str = None,
        enable_scheduler: bool = False,
    ):
        self.k = k
        self.b = b
        self.initial_b = b

        self.enable_scheduler = enable_scheduler

        self.output_dir = output_dir

        self.ave_target_rank = target_rank 
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        # breakpoint()
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 


    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                #  Check the shape difference between exp_avg_ipt and current parameter 
                if self.exp_avg_ipt[n].shape != p.shape:
                    new_shape = list(p.shape)
                    old_shape = list(self.exp_avg_ipt[n].shape)
                    
                    # for each dimension check whether an expansion is needed
                    padding = []
                    for i in range(len(new_shape) - 1, -1, -1):
                        if old_shape[i] < new_shape[i]:
                            padding_size = new_shape[i] - old_shape[i]
                            padding.extend([0, padding_size])
                        else:
                            padding.extend([0, 0])
                    
                    # Pad each dimension that requires expansion
                    self.exp_avg_ipt[n] = torch.nn.functional.pad(self.exp_avg_ipt[n], tuple(padding), "constant", 0)
                    self.exp_avg_unc[n] = torch.nn.functional.pad(self.exp_avg_unc[n], tuple(padding), "constant", 0)
                with torch.no_grad():
                    # Calculate sensitivity 
                    try:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    except:
                        breakpoint()
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                    

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank):
        # breakpoint()
        is_dict = {}
        combine_dict = {} 
        singular_dict = {}

        lora_A_list = []
        lora_B_list = []
        lora_E_list = []
        # Calculate the importance score for each sub matrix 
        for n, p in model.named_parameters(): 
            if "lora_A" in n: 
                lora_A_list.append(p)
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                combine_dict.setdefault(name_mat, []).append(comb_ipt)
            elif "lora_B" in n: 
                lora_B_list.append(p)
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                combine_dict.setdefault(name_mat, []).append(comb_ipt)
            elif "lora_E" in n:
                lora_E_list.append(p)
                ipt_score = self.calculate_score(n, p=p, metric="ipt")                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores 
        all_is = []
        for name_mat in combine_dict: 
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))
        
            
        
        # if self.output_dir:
        #     log_file = os.path.join(self.output_dir, "importance_log.txt")
        #     os.makedirs(self.output_dir, exist_ok=True)
            
        #     with open(log_file, 'a') as f:
        #         f.write(f"Step {self.global_step}:\n")
        #         f.write(f"{all_is}\n\n")
        
        # breakpoint()
        


        top_k_elements = []
        sublist_sizes = [] # how many elements are picked from each sublist
        
        for sublist in all_is:
            k = int(min(self.k, sublist.numel() - 1)) # prevent deleting all elements
            top_k_elements.append(torch.topk(sublist, k, largest=False).values)
            sublist_sizes.append(k)
        
        

        flat_top_k_elements = torch.cat(top_k_elements)
        smallest_b_elements = torch.topk(flat_top_k_elements, self.b, largest=False).values
        largest_b_elements = torch.topk(flat_top_k_elements, self.b, largest=True).values

        mask_threshold = smallest_b_elements.max().item()

        decrease_idx = torch.topk(flat_top_k_elements, self.b, largest=False).indices
        increase_idx = torch.topk(flat_top_k_elements, self.b, largest=True).indices
        
        # breakpoint()
        
        
        ### Randomly pick ###
        # increase_idx = torch.randperm(flat_top_k_elements.size(0))[:self.b]

        ###############
        
        
        
        ########### Experiments ###############
        # global_averaged_all_is =  [0.5*tensor + 0.5*tensor.mean() for tensor in all_is]
        # top_k_elements_expand = []
        # sublist_sizes_expand = []
        
        # for sublist in global_averaged_all_is:
        #     k = min(self.k, sublist.numel() - 1) # prevent deleting all elements
        #     top_k_elements_expand.append(torch.topk(sublist, k, largest=False).values)
        #     sublist_sizes_expand.append(k)
            
        
        # flat_top_k_elements_expand = torch.cat(top_k_elements_expand)
        # increase_idx = torch.topk(flat_top_k_elements_expand, self.b, largest=True).indices
         ########### Experiments ###############


        def map_indices(flat_indices, sublist_sizes):
            """
            sublist_sizes: [2, 1, 2] -> [[0,1], [2], [3, 4]]
            flat_indices: [1, 3]
            return: [0, 2]
            """
            mapped_sublist_ids = []
            current_offset = 0
            for sublist_id, size in enumerate(sublist_sizes):
                for idx in flat_indices:
                    if current_offset <= idx < current_offset + size:
                        mapped_sublist_ids.append(sublist_id)
                current_offset += size
            return mapped_sublist_ids
        
        decrease_idx = map_indices(decrease_idx, sublist_sizes)
        increase_idx = map_indices(increase_idx, sublist_sizes)
        
        ########### Experiments ###############
        # increase_idx = map_indices(increase_idx, sublist_sizes_expand)
        ########### Experiments ###############

        # Mask out unimportant singular values 
        # with torch.no_grad():
        #     curr_sum_rank = 0
        #     sum_param = 0
            
            
        #     for n, p in model.named_parameters():
        #         if "lora_E" in n: 
        #             p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
                    # ranknum = (is_dict[n]>mask_threshold).sum().item() 

                    # if self.tb_writter is not None and self.global_step%self.log_interval==0:
                    #     self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                    #     self.rank_pattern[n] = ranknum 
                    #     curr_sum_rank += ranknum 
                    #     sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                    #     sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_B")][0]



        def set_nested_attr(obj, attr, value):
            # Set the value of a nested attribute of an object
            # obj here will be the model
            attrs = attr.split('.')
            for attr_name in attrs[:-1]:
                obj = getattr(obj, attr_name)
            setattr(obj, attrs[-1], value)

       
        num_matrix = len(lora_A_list)
        
        # Map each matrix in lora_E_list to its corresponding name in is_dict
        lora_E_name_map = {p: name for name, p in model.named_parameters() if "lora_E" in name}


 ################## Decrease the rank of the matrix and update model parameters  ##################
        for i in range(num_matrix):
            if i in decrease_idx:
                # if i in increase_idx:
                #     continue
                
                matrix_A = lora_A_list[i]  # (r, hdim_a)
                matrix_B = lora_B_list[i]  # (hdim_b, r)
                matrix_E = lora_E_list[i]  # (r, 1)
                
                # Adjusting the size of new_vector to match matrix_A's second dimension
                with torch.no_grad():
                   
                    matrix_E_name = lora_E_name_map[matrix_E]
                    matrix_A_name = matrix_E_name.replace("lora_E", "lora_A")
                    matrix_B_name = matrix_E_name.replace("lora_E", "lora_B")
                    keep_indices = (is_dict[matrix_E_name] > mask_threshold).nonzero(as_tuple=True)[0]
                    
                    # Get the ranks below the threshold
                    importance_scores = is_dict[matrix_E_name]
                    below_threshold_indices = (importance_scores <= mask_threshold).nonzero(as_tuple=True)[0]
                    # Convert scores to 1D if necessary
                    below_threshold_scores = importance_scores[below_threshold_indices].squeeze()  # Ensure 1D

                    # Limit the number of ranks to be removed based on sublist_sizes
                    num_to_remove = min(sublist_sizes[i], below_threshold_scores.numel())
                    if num_to_remove > 0:
                        # Select the indices with the lowest scores
                        removal_indices = torch.topk(
                            below_threshold_scores, 
                            num_to_remove, 
                            largest=False
                        ).indices
                        # Map back to the original indices
                        removal_indices = below_threshold_indices[removal_indices]
                    else:
                        # Nothing to remove
                        removal_indices = torch.tensor([], dtype=torch.long, device=importance_scores.device)
                    
                    # Compute keep_indices (all indices excluding removal_indices)
                    keep_indices = torch.arange(importance_scores.numel(), device=importance_scores.device)
                    keep_indices = torch.tensor(
                        [idx for idx in keep_indices if idx not in removal_indices], 
                        dtype=torch.long, 
                        device=importance_scores.device
                    )

                    # Prune matrix_A, matrix_B, and matrix_E based on keep_indices
                    pruned_matrix_A = torch.index_select(matrix_A, 0, keep_indices)  # Select only rows in A to keep
                    pruned_matrix_B = torch.index_select(matrix_B, 1, keep_indices)  # Select only columns in B to keep
                    pruned_matrix_E = torch.index_select(matrix_E, 0, keep_indices)  # Select only elements in E to keep
                    
                    # Prune importance tracking variables for lora_A, lora_B, and lora_E
                    if matrix_E_name in self.ipt:
                        self.ipt[matrix_E_name] = torch.index_select(self.ipt[matrix_E_name], 0, keep_indices)
                    if matrix_A_name in self.ipt:
                        self.ipt[matrix_A_name] = torch.index_select(self.ipt[matrix_A_name], 0, keep_indices)
                    if matrix_B_name in self.ipt:
                        self.ipt[matrix_B_name] = torch.index_select(self.ipt[matrix_B_name], 1, keep_indices)

                    if matrix_E_name in self.exp_avg_ipt:
                        self.exp_avg_ipt[matrix_E_name] = torch.index_select(self.exp_avg_ipt[matrix_E_name], 0, keep_indices)
                    if matrix_A_name in self.exp_avg_ipt:
                        self.exp_avg_ipt[matrix_A_name] = torch.index_select(self.exp_avg_ipt[matrix_A_name], 0, keep_indices)
                    if matrix_B_name in self.exp_avg_ipt:
                        self.exp_avg_ipt[matrix_B_name] = torch.index_select(self.exp_avg_ipt[matrix_B_name], 1, keep_indices)

                    if matrix_E_name in self.exp_avg_unc:
                        self.exp_avg_unc[matrix_E_name] = torch.index_select(self.exp_avg_unc[matrix_E_name], 0, keep_indices)
                    if matrix_A_name in self.exp_avg_unc:
                        self.exp_avg_unc[matrix_A_name] = torch.index_select(self.exp_avg_unc[matrix_A_name], 0, keep_indices)
                    if matrix_B_name in self.exp_avg_unc:
                        self.exp_avg_unc[matrix_B_name] = torch.index_select(self.exp_avg_unc[matrix_B_name], 1, keep_indices)



                    # Convert pruned matrices to parameters and update the model
                    pruned_matrix_A = torch.nn.Parameter(pruned_matrix_A)
                    pruned_matrix_B = torch.nn.Parameter(pruned_matrix_B)
                    pruned_matrix_E = torch.nn.Parameter(pruned_matrix_E)

                    lora_A_list[i] = pruned_matrix_A
                    lora_B_list[i] = pruned_matrix_B
                    lora_E_list[i] = pruned_matrix_E

                # Replace pruned matrices in the model
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param is matrix_A:
                            set_nested_attr(model, name, pruned_matrix_A)
                        elif param is matrix_B:
                            set_nested_attr(model, name, pruned_matrix_B)
                        elif param is matrix_E:
                            set_nested_attr(model, name, pruned_matrix_E)
                
 ################## Increase the rank of the matrix and update model parameters  ################## 
        for i in increase_idx:
            matrix_A = lora_A_list[i]  # (r, hdim_a)
            matrix_B = lora_B_list[i]  # (hdim_b, r)
            matrix_E = lora_E_list[i]  # (r, 1)

            # Adjusting the size of new_vector to match matrix_A's second dimension
            with torch.no_grad():
                new_vector = torch.randn(matrix_A.size(1), device=matrix_A.device, requires_grad=True)
                new_vector = new_vector - matrix_A.T @ (matrix_A @ new_vector)
                new_vector = new_vector / (new_vector.norm() + 1e-6)
                new_matrix_A = torch.cat([matrix_A, new_vector.unsqueeze(0)], dim=0)
                new_matrix_A = torch.nn.Parameter(new_matrix_A)
                lora_A_list[i] = new_matrix_A #! update the lora_A_list
                
                

            # Replace the parameter in the model by matching the parameter tensor directly
            with torch.no_grad():
                # for param in model.parameters():
                #     if param is matrix_A:
                #         # import pdb; pdb.set_trace()
                #         param.data = new_matrix_A
                for name, param in model.named_parameters():
                    if param is matrix_A:
                        set_nested_attr(model, name, new_matrix_A)  # This is a hacky way to update the model parameter


            # Adjusting the size of new_vector to match matrix_B's first dimension
            with torch.no_grad():
                new_vector = torch.randn(matrix_B.size(0), device=matrix_B.device, requires_grad=True)
                new_vector = new_vector - matrix_B @ (matrix_B.T @ new_vector)
                new_vector = new_vector / (new_vector.norm() + 1e-6)
                new_matrix_B = torch.cat([matrix_B, new_vector.unsqueeze(1)], dim=1)
                new_matrix_B = torch.nn.Parameter(new_matrix_B)
                lora_B_list[i] = new_matrix_B #! update the lora_B_list

            # Replace the parameter in the model by matching the parameter tensor directly
            with torch.no_grad():
                # for param in model.parameters():
                #     if param is matrix_B:
                #         # import pdb; pdb.set_trace()
                #         param.data = new_matrix_B
                for name, param in model.named_parameters():
                    if param is matrix_B:
                        set_nested_attr(model, name, new_matrix_B)

            # Adjust matrix_E
            with torch.no_grad():
                new_scalar = torch.tensor([min(matrix_E.view(-1).abs().min().item(), 1e-13)], device=matrix_E.device, requires_grad=True)
                new_matrix_E = torch.cat([matrix_E, new_scalar.unsqueeze(0)], dim=0)
                new_matrix_E = torch.nn.Parameter(new_matrix_E)
                lora_E_list[i] = new_matrix_E #! update the lora_E_list
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param is matrix_E:
                        # import pdb; pdb.set_trace()
                        set_nested_attr(model, name, new_matrix_E)
            
        # # record impotrance score trend
        # if self.tb_writter is not None:
        #     # Create a directory for the importance score plots
        #     ipt_dir = os.path.join(self.output_dir, "ipt_plots")
        #     os.makedirs(ipt_dir, exist_ok=True)
        #     image_path = os.path.join(ipt_dir, f"step_{self.global_step}.png")
        #     plot_ipt_graph(all_is, image_path)
        
        ipt_score_boolean = True
        # Save importance scores
        if ipt_score_boolean:
            # Create a directory for the importance score plots
            ipt_dir = os.path.join(self.output_dir, "ipt_scores")
            os.makedirs(ipt_dir, exist_ok=True)
            
            ipt_score_path = os.path.join(ipt_dir, f"step_{self.global_step}.json")
            with open(ipt_score_path, "w") as file:
                all_is_serializable = [item.tolist() if isinstance(item, torch.Tensor) else item for item in all_is]
                json.dump(all_is_serializable, file)

        # record ranknum
        if self.tb_writter is not None:                    
            for n, p in model.named_parameters():
                if "lora_E" in n: 
                    # ranknum = (is_dict[n]!=0.0).sum().item() 
                    ranknum = (p != 0.0).sum().item()
                    # print(n,p)
                    # print("\n")
                    self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                    self.rank_pattern[n] = ranknum
            # print(self.rank_pattern)
            # Define the directory path
            rank_distribution_dir = os.path.join(self.output_dir, "rank_plots")
            os.makedirs(rank_distribution_dir, exist_ok=True)
            image_path = os.path.join(rank_distribution_dir, f"step_{self.global_step}.png")
            
            plotting_global_max = max(10, self.lora_init_rank*2)    
            plot_rank(self.rank_pattern, image_path, 1, plotting_global_max)
                            

        return mask_threshold




    def update_and_mask(self, model, global_step):
        self.global_step=global_step
        mask_threshold = None
        # if self.initial_warmup < global_step < self.total_step-self.final_warmup:
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model)
            # Budget schedule
            if self.enable_scheduler:
                # print("[Scheduler] Now is enabled")
                self._b_scheduler(global_step)
            if global_step > self.initial_warmup and (global_step-self.initial_warmup) % self.mask_interval == 0 and self.b > 0:
                print(f"[Scheduler] Now is masking, b={self.b}")
                mask_threshold = self.mask_to_target_rank(model, 0)
                
        return 0, mask_threshold
    
    def _b_scheduler(self, global_step):
        initial_b = self.initial_b
        final_b = 0
        total_step = self.total_step
        progress = (global_step - self.initial_warmup) / (total_step - self.final_warmup - self.initial_warmup)
        progress = min(max(progress, 0), 1)
        mul_coeff = progress ** 3
        self.b = round(initial_b + (final_b - initial_b) * mul_coeff)




def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param