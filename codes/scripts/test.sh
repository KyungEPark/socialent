#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu-single
#SBATCH --time=00:30:00
#SBATCH --mem=200gb
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=test_park
#SBATCH --output /home/ma/ma_ma/ma_kyupark/socialent/logs/slurm-%j.out


now=$(date +"%T")

echo "Program starts:  $now"

source /home/ma/ma_ma/ma_kyupark/.bashrc
conda_initialize
micromamba activate is809
cd /home/ma/ma_ma/ma_kyupark/socialent

python codes/hfprompting.py --model microsoft/Phi-3-medium-4k-instruct --savefile Phi-3-medium-4k-instruct_reasons.parquet

end=$(date +"%T")
echo "Completed: $end"
