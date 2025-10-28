#!/bin/bash -x
#SBATCH --account=westai0091           # Account details
#SBATCH --nodes=1                        # Number of compute nodes required
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --gres=gpu:1                  
#SBATCH --time=05:00:00                  # Maximum runtime
#SBATCH --mem=50gb
#SBATCH --partition=dc-hwai
#SBATCH --output /p/project/westai0091/socialent/logs/slurm-%j.out

now=$(date +"%T")

echo "Program starts:  $now"

source /home/ma/ma_ma/ma_kyupark/.bashrc
conda_initialize
micromamba activate is809
cd /gpfs/bwfor/work/ws/ma_kyupark-socent/socialent

python codes/story_reason.py --model Qwen/Qwen3-8B --savefile Qwen/Qwen3-8B_pred_reason.parquet

end=$(date +"%T")
echo "Completed: $end"
