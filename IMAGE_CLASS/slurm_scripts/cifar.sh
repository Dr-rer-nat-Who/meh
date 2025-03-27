#!/bin/bash
#SBATCH --job-name=run_vit_cifar
#SBATCH -c 4                     # 分配 4 个 CPU 核心
#SBATCH -N 1                     # 只使用 1 个节点
#SBATCH -t 12:00:00               # 运行时间限制为 12 小时
#SBATCH -p seas_gpu              # 选择 GPU 分区
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1  # 申请 1 块 NVIDIA A100 GPU
#SBATCH --mem=256GB              
#SBATCH -o /n/home08/zicheng/ElaLoRA/IMAGE_CLASS/run_vit_cifar_%j.out
#SBATCH -e /n/home08/zicheng/ElaLoRA/IMAGE_CLASS/run_vit_cifar_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=zichengma@g.harvard.edu

# Change to script directory
cd /n/home08/zicheng/ElaLoRA/IMAGE_CLASS
export WANDB_DISABLED=true
# Load environment
source activate IMAGE
which python
python -c 'print("Hi Zicheng. Your job is running!")'

# 启动 run_vit_cifar.sh
bash /n/home08/zicheng/ElaLoRA/IMAGE_CLASS/bash_scripts/cifar.sh

python -c 'print("Hi Zicheng. Everything is done!")'
