#!/bin/bash
#SBATCH --job-name=bib_tom # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kvg245@nyu.edu    # Where to send mail
#SBATCH --nodes=1                    # Run all processes on  a single node
#SBATCH --ntasks=12                  # Number of proces ses
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mem=100GB                   # Total memory limit
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --output=%j.log # Standard output and error log

cd /data/kvg245/bib-tom
source venv/bin/activate
git pull
# python3 train_tom.py --data_path /misc/vlgscratch4/LakeGroup/kanishk/bg_train/ --types coe pre mue see --seed 1 --batch_size 32 --max_epochs 1000 --gpus 1 --num_workers 12  --stochastic_weight_avg True --lr 1e-4 --check_val_every_n_epoch 1 --max_epochs 1000 --process_data 0 --track_grad_norm 2 --gradient_clip_val 0.5 
python3 train_tom.py --data_path /misc/vlgscratch4/LakeGroup/kanishk/bg_train/ --types coe pre mue see --seed 1 --batch_size 64 --max_epochs 1000 --gpus 1 --num_workers 12  --stochastic_weight_avg False --lr 1e-5 --check_val_every_n_epoch 1 --max_epochs 1000 --process_data 0 --track_grad_norm 2 --gradient_clip_val 0.5 --resume_from_checkpoint /data/kvg245/bib-tom/lightning_logs/version_952410/checkpoints/epoch\=63-step\=22399.ckpt
