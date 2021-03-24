#!/bin/bash
#SBATCH --job-name=bib_tom # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kvg245@nyu.edu    # Where to send mail
#SBATCH --nodes=1                    # Run all processes on  a single node
#SBATCH --ntasks=16                  # Number of proces ses
#SBATCH --gpus=1
#SBATCH --mem=100GB                   # Total memory limit
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --output=%j.log # Standard output and error log

cd /data/kvg245/bib-tom
source venv/bin/activate
git pull
python3 train_tom.py --data_path /misc/vlgscratch4/LakeGroup/kanishk/bg_train/aug280k --types co pr --batch_size 512 --max_epochs 100 --val_check_interval 1 --gpus 1 --gradient_clip_val 10 --num_workers 16 --track_grad_norm 2 --num_sanity_val_steps 2 --stochastic_weight_avg True
