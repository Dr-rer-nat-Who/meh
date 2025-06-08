import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForImageClassification
from datasets import load_dataset
from dataset_vtab import get_data
from safetensors.torch import load_file  # pip install safetensors
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

############################################
# 模型加载相关函数：动态更新模型结构
############################################

def get_module_by_name(model, module_path):
    """根据类似 'vit.encoder.layer.0.attention.output.dense' 的模块路径返回对应模块"""
    attrs = module_path.split('.')
    mod = model
    try:
        for attr in attrs:
            if attr.isdigit():
                mod = mod[int(attr)]
            else:
                mod = getattr(mod, attr)
        return mod
    except AttributeError as e:
        print(f"Error accessing module {module_path}: {e}")
        return None

def update_module_lora(module, param_name, loaded_tensor):
    """
    对指定模块中 lora 参数进行更新：如果尺寸不匹配，则直接用 checkpoint 中的参数替换，
    同时更新模块内部的 rank 属性（这里以 lora_A/lora_E 使用第一维、lora_B 使用第二维为例）。
    """
    current_tensor = getattr(module, param_name)
    if current_tensor.shape != loaded_tensor.shape:
        print(f"Updating module {module} parameter {param_name}: model shape {current_tensor.shape} -> checkpoint shape {loaded_tensor.shape}.")
        new_param = nn.Parameter(loaded_tensor)
        setattr(module, param_name, new_param)
        # 更新 rank 属性示例（请根据你的实际 ElaLoRA 模块设计调整）
        if param_name in ["lora_A", "lora_E"]:
            module.rank = loaded_tensor.shape[0]
        elif param_name == "lora_B":
            module.rank = loaded_tensor.shape[1]
    else:
        print(f"Module {module} parameter {param_name} shape matches.")

def update_model_structure(model, loaded_sd):
    """
    遍历 loaded_sd 中所有包含 "lora" 的参数，动态定位对应模块并检查尺寸，
    如有不匹配则修改模型中对应模块的结构（即直接替换参数并更新 rank）。
    """
    for key, loaded_param in loaded_sd.items():
        if "lora" not in key:
            continue
        # 例如 key: "vit.encoder.layer.0.attention.output.dense.lora_A"
        parts = key.split('.')
        param_name = parts[-1]
        module_path = '.'.join(parts[:-1])
        module = get_module_by_name(model, module_path)
        if module is None:
            print(f"Module {module_path} not found, skipping key {key}.")
            continue
        if not hasattr(module, param_name):
            print(f"Module {module_path} does not have attribute {param_name}, skipping.")
            continue
        update_module_lora(module, param_name, loaded_param)

def load_checkpoint(model, checkpoint_dir):
    """
    从 checkpoint 目录加载模型配置，并利用 safetensors 加载模型权重，然后更新模型结构，
    最后将权重载入模型中。
    """
    # 1. 从 checkpoint 目录加载配置
    config = AutoConfig.from_pretrained(checkpoint_dir)
    # 2. 根据配置创建模型实例（不自动加载权重）
    model = AutoModelForImageClassification.from_config(config)
    # 3. 加载 safetensors 格式的模型权重
    safetensors_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(safetensors_file):
        raise ValueError(f"Model file {safetensors_file} not found!")
    print("Loading checkpoint weights from safetensors...")
    loaded_sd = load_file(safetensors_file)
    # 4. 动态更新模型结构以适应 checkpoint 中 lora 参数的尺寸
    update_model_structure(model, loaded_sd)
    # 5. 载入模型权重（允许缺失/多余）
    missing_keys, unexpected_keys = model.load_state_dict(loaded_sd, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    return model

############################################
# 测试函数：使用 avalanche Accuracy 进行评估
############################################

@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    pbar = tqdm(dl)
    model.cuda()
    for batch in pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        # breakpoint()
        outputs = model(x)
        # 直接使用 outputs.logits 获取预测结果
        logits = outputs.logits
        preds = logits.argmax(dim=1)
        acc.update(preds, y)

    return acc.result()

############################################
# 主函数：解析参数，加载模型，对每个子数据集运行 evaluation
############################################

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to the model checkpoint directory (safetensors format).')
    parser.add_argument('--vtab_root', type=str, required=True,
                        help='Path to the VTAB-1k dataset root directory, which contains sub-datasets.')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    print(args)
    
    # 1. 载入修改好的模型
    print("Loading model from checkpoint...")
    model = load_checkpoint(None, args.checkpoint_dir)
    
    # 2. 遍历 VTAB 根目录下的所有子数据集进行评估
    vtab_root = "/n/netscratch/kung_lab/Everyone/zma/vtab-1k"
    subdatasets = sorted([d for d in os.listdir(vtab_root) if os.path.isdir(os.path.join(vtab_root, d))])
    eval_results = {}

    for sub in subdatasets:
        sub_dir = os.path.join(vtab_root, sub)
        print(f"\nEvaluating sub-dataset: {sub}")
        # 使用 load_dataset("imagefolder", data_dir=...) 直接加载该子数据集
        _, test_dl = get_data(sub_dir)

        
        # 调用测试函数评估该子数据集
        accuracy = test(model, test_dl)
        eval_results[sub] = accuracy
        print(f"Sub-dataset {sub} Accuracy: {accuracy:.4f}")
    
    print("\n===== All Evaluation Results =====")
    total_acc = 0
    for sub, acc in eval_results.items():
        total_acc += acc
        print(f"{sub}: Accuracy = {acc:.4f}")

    print(f"Avg accuracy is: {total_acc/len(eval_results.items())}")
