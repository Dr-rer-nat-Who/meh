#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

from typing import Dict

from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    
# def plot_rank(data, file_path):
#     layers = sorted(set(int(k.split(".")[3]) for k in data.keys()))
#     weights = sorted(set(".".join(k.split(".")[4:-1]) for k in data.keys()))

#     heatmap_data = pd.DataFrame(index=weights, columns=layers)

#     for key, value in data.items():
#         layer = int(key.split(".")[3])
#         weight = ".".join(key.split(".")[4:-1])
#         heatmap_data.loc[weight, layer] = value

#     heatmap_data = heatmap_data.astype(float)  # Ensure numeric values

#     # Plot the heatmap
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".0f", cbar_kws={'label': 'Rank'})
#     plt.title("DeBERTa Rank Heatmap")
#     plt.xlabel("Layer")
#     plt.ylabel("Weight Matrix")
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(file_path, bbox_inches='tight')

def plot_rank(data, file_path, global_min=None, global_max=None):
    layers = sorted(set(int(k.split(".")[3]) for k in data.keys()))
    weights = sorted(set(".".join(k.split(".")[4:-1]) for k in data.keys()))

    heatmap_data = pd.DataFrame(index=weights, columns=layers)

    for key, value in data.items():
        layer = int(key.split(".")[3])
        weight = ".".join(key.split(".")[4:-1])
        heatmap_data.loc[weight, layer] = value

    heatmap_data = heatmap_data.astype(float)  # Ensure numeric values

    # Determine global min and max if not provided
    if global_min is None:
        global_min = heatmap_data.min().min()
    if global_max is None:
        global_max = heatmap_data.max().max()

    # Plot the heatmap with consistent color range
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        fmt=".0f",
        cbar_kws={'label': 'Rank'},
        vmin=global_min,
        vmax=global_max,
    )
    plt.title("Rank Heatmap")
    plt.xlabel("Layer")
    plt.ylabel("Weight Matrix")
    plt.tight_layout()
    # plt.show()
    plt.savefig(file_path, bbox_inches='tight')


def plot_ipt_graph(all_is, file_path):
    # Plot the importance score trends for each matrix
    sorted_all_is = [torch.sort(tensor, descending=True)[0] for tensor in all_is]
    max_len = max(tensor.size(0) for tensor in sorted_all_is)
    padded_all_is = [F.pad(tensor, (0, max_len - tensor.size(0))) for tensor in sorted_all_is]

    # Convert to numpy for plotting
    padded_all_is_np = [tensor.cpu().numpy() for tensor in padded_all_is]

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, scores in enumerate(padded_all_is_np):
        plt.plot(np.arange(max_len), scores, label=f'Matrix {i}')
    
    plt.xlabel('Rank Index')
    plt.ylabel('Importance Score')
    plt.title('Importance Score Trends for Each Matrix')
    plt.legend()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()