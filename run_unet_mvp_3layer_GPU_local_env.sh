#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --account=none   # account name (mandatory), if the job runs under a project then it'll be the project name, if not then it should =none
#SBATCH --job-name=unet_effb7_3layer   # some descriptive job name of your choice
#SBATCH --output=%x-%j_out.txt      # output file name will contain job name + job ID
#SBATCH --error=%x-%j_err.txt       # error file name will contain job name + job ID
#SBATCH --gres=gpu:1               # GPU per node
#SBATCH --partition=gpu            # GPU partition
#SBATCH --time=6-21:00:00          # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
####################### low mem-per-gpu, orginal is 48G ################################
#SBATCH --mem-per-gpu=24G          # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1                  # 2 nodes, with each node having a GPU
#SBATCH --ntasks=1                 # each GPU will handle 1 task. script needs multiparallel
################### low CPU per task, original is 24 ###################
#SBATCH --cpus-per-task=4       # MARS has 2 CPU in GPU node with 32 cores per CPU
#SBATCH --ntasks-per-node=1        # number of tasks to be launched on each allocated node

############# LOADING MODULES (optional) #############
module purge
source /opt/gridware/depots/996bcebb/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda init bash
conda activate deeplearn
echo "Hello from $SLURM_JOB_NODELIST"
echo "conda environment is $CONDA_DEFAULT_ENV"
############# MY CODE #############
echo $(lspci | grep -i nvidia)
# srun python3 /users/ad394h/Documents/segment_blood_vessels/scripts/sm_unet_efficientnet_2_class_1024px_FINAL.py
# srun python3 /users/ad394h/Documents/segment_blood_vessels/scripts/microvascular_proliferation/sm_unet_efficientnet_2_class_1024px_getOutput.py
srun python3 /users/ad394h/Documents/microvascular_proliferation/scripts/sm_unet_effb7_3layer256_2class_mvp.py

echo "this means code ran well"
############# END #############
conda deactivate
echo "conda default environment is $CONDA_DEFAULT_ENV" 
echo "$CONDA_PREFIX"
