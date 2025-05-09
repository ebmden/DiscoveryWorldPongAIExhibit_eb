#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=11
#SBATCH --output=pong_job.out
#SBATCH --job-name=firstJob


. /usr/local/anaconda3/bin/activate

singularity exec --nv /data/containers/msoe-tensorflow-20.07-tf2-py3.sif python3 -m pip install --user -r /home/bassoe/srdes/DiscoveryWorldPongAIExhibit/requirements.txt
singularity exec --nv /data/containers/msoe-tensorflow-20.07-tf2-py3.sif python3 /home/bassoe/srdes/DiscoveryWorldPongAIExhibit/reinforcement.py
singularity exec --nv /data/containers/msoe-tensorflow-20.07-tf2-py3.sif python3 /home/bassoe/srdes/DiscoveryWorldPongAIExhibit/visualizer.py