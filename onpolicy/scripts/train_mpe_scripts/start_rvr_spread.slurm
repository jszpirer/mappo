#!/bin/bash
# Submission script for Lyra
#SBATCH --time=4-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres="gpu:1"
#SBATCH --mem=64G
#SBATCH --partition=batch
#
#SBATCH --output=out/%j_%x.out
#
echo "Loading modules"
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module list
echo "Modules loaded"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

export CUDA_VISIBLE_DEVICES=1

echo "Localscratch dir"
echo $TMPDIR

export ScratchDir="${TMPDIR}/rvr_spread/"
mkdir -p ${ScratchDir}
cp -r ../../../../mappo/ ${ScratchDir}
cp -r ../../../../env-mappo/ ${ScratchDir}

cd ${ScratchDir}mappo/onpolicy/scripts/train_mpe_scripts/

# export PYTHONPATH="/home/users/j/s/jszpirer/mappo/"
export PYTHONPATH="${ScratchDir}mappo/"

source ${ScratchDir}env-mappo/bin/activate

srun train_mpe_rvr_spread.sh

mkdir /home/users/j/s/jszpirer/rvrspreadwallscells10agents/seed2

scp -r ${ScratchDir}mappo/onpolicy/scripts/results/MPE/rvr_spread_local_omni_walls_fast_cells/rmappo/rvr_local_omni_12and5both/wandb/ /home/users/j/s/jszpirer/rvrspreadwallscells10agents/seed2

