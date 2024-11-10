import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer 
from typing import Optional, List

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
        b: int = 2,
    ):
        self.k = k
        self.b = b

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
                                # 检查当前参数和 exp_avg_ipt 的形状差异
                                # 检查当前参数和 exp_avg_ipt 的形状差异
                if self.exp_avg_ipt[n].shape != p.shape:
                    new_shape = list(p.shape)
                    old_shape = list(self.exp_avg_ipt[n].shape)
                    
                    # 针对每个维度判断是否需要扩展
                    padding = []
                    for i in range(len(new_shape) - 1, -1, -1):
                        if old_shape[i] < new_shape[i]:
                            padding_size = new_shape[i] - old_shape[i]
                            padding.extend([0, padding_size])
                        else:
                            padding.extend([0, 0])
                    
                    # 将所有需要扩展的维度进行填充
                    self.exp_avg_ipt[n] = torch.nn.functional.pad(self.exp_avg_ipt[n], tuple(padding), "constant", 0)
                    self.exp_avg_unc[n] = torch.nn.functional.pad(self.exp_avg_unc[n], tuple(padding), "constant", 0)
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach()
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

        top_k_elements = torch.stack([torch.topk(sublist, self.k, largest=False).values for sublist in all_is])
        smallest_b_elements = torch.topk(top_k_elements.view(-1), self.b, largest=False).values
        mask_threshold = smallest_b_elements.max().item()
        largest_b_elements = torch.topk(top_k_elements.view(-1), self.b, largest=True).values
        increase_idx = torch.topk(top_k_elements.view(-1), self.b, largest=True).indices
        increase_idx = [(idx // self.k).item() for idx in increase_idx]

        # Mask out unimportant singular values 
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_E" in n: 
                    p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)



        def set_nested_attr(obj, attr, value):
            # Set the value of a nested attribute of an object
            # obj here will be the model
            attrs = attr.split('.')
            for attr_name in attrs[:-1]:
                obj = getattr(obj, attr_name)
            setattr(obj, attrs[-1], value)

        # Increase the rank of the matrix and update model parameters
        num_matrix = len(lora_A_list)
        for i in range(num_matrix):
            if i in increase_idx:
                matrix_A = lora_A_list[i]  # (r, hdim_a)
                matrix_B = lora_B_list[i]  # (hdim_b, r)
                matrix_E = lora_E_list[i]  # (r, 1)
                # breakpoint()
                # Adjusting the size of new_vector to match matrix_A's second dimension
                with torch.no_grad():
                    new_vector = torch.randn(matrix_A.size(1), device=matrix_A.device, requires_grad=True)
                    new_vector = new_vector - matrix_A.T @ (matrix_A @ new_vector)
                    new_vector = new_vector / (new_vector.norm() + 1e-6)
                    new_matrix_A = torch.cat([matrix_A, new_vector.unsqueeze(0)], dim=0)
                    new_matrix_A = torch.nn.Parameter(new_matrix_A)

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

                # Replace the parameter in the model by matching the parameter tensor directly
                with torch.no_grad():
                    # for param in model.parameters():
                    #     if param is matrix_B:
                    #         breakpoint()
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

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param is matrix_E:
                            # import pdb; pdb.set_trace()
                            set_nested_attr(model, name, new_matrix_E)
                            

        return mask_threshold




    def update_and_mask(self, model, global_step):
        mask_threshold = None
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model)
            # Budget schedule
            if global_step % self.mask_interval == 0:
                mask_threshold = self.mask_to_target_rank(model, 0)
                
        return 0, mask_threshold


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