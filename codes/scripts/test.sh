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

source /p/project/westai0091/.bashrc
conda_initialize
micromamba activate stat
cd /p/project/westai0091/socialent

python codes/hfprompting.py --model microsoft/Phi-3-medium-4k-instruct --savefile Phi-3-medium-4k-instruct_reasons.parquet

end=$(date +"%T")
echo "Completed: $end"
