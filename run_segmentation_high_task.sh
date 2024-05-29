#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --account=none   # account name (mandatory), if the job runs under a project then it'll be the project name, if not then it should =none
#SBATCH --job-name=run_segmentation   # some descriptive job name of your choice
#SBATCH --output=%x-%j_out.txt      # output file name will contain job name + job ID
#SBATCH --error=%x-%j_err.txt       # error file name will contain job name + job ID
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu         # which partition to use, default on MARS is “nodes"
#SBATCH --time=0-06:00:00       # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=4G                # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1              # number of nodes to allocate, default is 1
#SBATCH --ntasks=1         # number of Slurm tasks to be launched, increase for multi-process runs ex. SPECIFY even for the GPU
#SBATCH --cpus-per-task=4     # number of processor cores to be assigned for each task, default is 1, increase for multi-threaded runs or GPU
#SBATCH --ntasks-per-node=1     # number of tasks to be launched on each allocated node

############# LOADING MODULES (optional) #############
module purge
module load apps/anaconda3/2023.03/bin # it has opencv and torch and segmentation_models_pytorch
module load apps/nvidia-cuda
#source /users/ad394h/miniforge-pypy3/etc/profile.d/mamba.sh
#mamba activate segmentation
############# MY CODE #############
echo "Hello from $SLURM_JOB_NODELIST"
# python3 /users/ad394h/Documents/scripts/tumor_normal_digital_brain_tumor_final.py
python3 /users/ad394h/Documents/segment_blood_vessels/src/histosegnet_base_model.py
echo "this means code ran well"

#mamba deactivate 
############# END #############
