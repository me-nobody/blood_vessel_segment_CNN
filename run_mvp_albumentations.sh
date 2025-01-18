#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --account=none   # account name (mandatory), if the job runs under a project then it'll be the project name, if not then it should =none
#SBATCH --job-name=albumentations   # some descriptive job name of your choice
#SBATCH --output=%x-%j_out.txt      # output file name will contain job name + job ID
#SBATCH --error=%x-%j_err.txt      # error file name will contain job name + job ID
#SBATCH --partition=nodes        # which partition to use, default on MARS is “nodes"
#SBATCH --time=0-01:00:00       # time limit for the whole run, in the form of d-hh:mm:ss, also accepts mm, mm:ss, hh:mm:ss, d-hh, d-hh:mm
#SBATCH --mem=16G                # memory required per node, in the form of [num][M|G|T]
#SBATCH --nodes=1               # number of nodes to allocate, default is 1
#SBATCH --ntasks=4              # number of Slurm tasks to be launched, increase for multi-process runs ex. MPI
#SBATCH --cpus-per-task=8       # number of processor cores to be assigned for each task, default is 1, increase for multi-threaded runs
#SBATCH --ntasks-per-node=4     # number of tasks to be launched on each allocated node

############# LOADING MODULES (optional) #############
module purge
source /opt/gridware/depots/996bcebb/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda init bash
conda activate deeplearn
echo "Hello from $SLURM_JOB_NODELIST"
echo "conda environment is $CONDA_DEFAULT_ENV"
############# MY CODE #############
python3 /users/ad394h/Documents/microvascular_proliferation/scripts/images_augment_microvascular_proliferation.py
############# END #################
echo "over with job"
conda deactivate
echo "conda default environment is $CONDA_DEFAULT_ENV" 
echo "$CONDA_PREFIX"

