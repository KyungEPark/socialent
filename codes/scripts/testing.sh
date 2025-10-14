#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu-single
#SBATCH --time=01:00:00
#SBATCH --mem=200gb
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=trans_reason
#SBATCH --output /gpfs/bwfor/work/ws/ma_kyupark-socent/socialent/logs/slurm-%j.out


now=$(date +"%T")

echo "Program starts:  $now"

source /home/ma/ma_ma/ma_kyupark/.bashrc
conda_initialize
micromamba activate is809
cd /gpfs/bwfor/work/ws/ma_kyupark-socent/socialent

python codes/test_reason.py --model Qwen/Qwen3-8B --savefile Qwen/Qwen3-8B_pred_reason.parquet

end=$(date +"%T")
echo "Completed: $end"
