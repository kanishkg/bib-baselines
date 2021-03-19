#!/bin/bash
#SBATCH --job-name=bib_cache # Job name
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

python3 cache_atc.py --data_path /misc/vlgscratch4/LakeGroup/kanishk/bg_train/ --types co pr --size 84 --ckpt /data/kvg245/bib-tom/lightning_logs/version_909749/checkpoints/epoch\=8-step\=110779.ckpt --cache_file noaug150k --gpus 1 --hparams /data/kvg245/bib-tom/lightning_logs/version_909749/hparams.yaml
