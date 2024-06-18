#!/bin/bash
#SBATCH -J gs-slam-gtri-jetski                                               # Job name
#SBATCH --account=gts-ychen3538                                 # Tracking account
#SBATCH -N1 --gres=gpu:A100:1                                   # Number of nodes and cores per node required
#SBATCH -t720                                                   # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                               # QOS Name
#SBATCH -o gs-slam-gtri-jetski.out                                   # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                              # Mail preferences
#SBATCH --mail-user=ichadha3@gatech.edu                         # E-mail address for notifications

cd /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/monogs-no-o3d/MonoGS

module load anaconda3/2022.05                                   # Load module dependencies
module load cuda/12.1

conda activate MonoGS

python slam.py --config configs/mono/gtri/jetski.yaml