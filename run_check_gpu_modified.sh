#!/bin/bash -l

############# SLURM SETTINGS #############
#SBATCH --account=none   
#SBATCH --job-name=check_gpu_resources   
#SBATCH --output=%x-%j_out.txt      
#SBATCH --error=%x-%j_err.txt       
#SBATCH --gres=gpu:a40:1 
#SBATCH --partition=gpu            
#SBATCH --time=0-06:00:00          
#SBATCH --mem-per-gpu=1G           
#SBATCH --nodes=1              
#SBATCH --ntasks=4             
#SBATCH --cpus-per-task=4     
#SBATCH --ntasks-per-node=4    

############# LOADING MODULES (optional) #############
#module purge
source /opt/gridware/depots/996bcebb/el7/pkg/apps/anaconda3/2023.03/bin/etc/profile.d/conda.sh
conda init bash
conda activate /mnt/scratch/users/ad394h/sharedscratch/anu
echo "Hello from $SLURM_JOB_NODELIST"
echo "conda environment is $CONDA_DEFAULT_ENV"
############################################################################
python3 /users/ad394h/Documents/scripts/check_GPU_resources.py

echo "this means code ran"
echo $(lspci | grep -i nvidia)

############# END #############
conda deactivate

echo "conda default environment is $CONDA_DEFAULT_ENV" 

echo "$CONDA_PREFIX" 
