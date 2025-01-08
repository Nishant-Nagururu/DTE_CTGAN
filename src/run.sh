#!/bin/bash
#SBATCH --job-name=CTGAN
#SBATCH --output=/blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/hpg_outputs/vasp2.out
#SBATCH --error=/blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/hpg_outputs/vasp2.err
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nishant.nagururu@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=100000
#SBATCH --time=48:00:00

# Load required modules
cd /blue/dream_team/CT_GAN_TEAM/CTGAN
module unload
module load cuda/11.4.3 nvhpc/23.7 openmpi/4.1.5 vasp/6.4.1 tensorflow
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/apps/compilers/nvhpc/23.7/Linux_x86_64/23.7/cuda"

# rm -rf /blue/dream_team/CT_GAN_TEAM/CTGAN/.venv
python -m venv .venv
source /blue/dream_team/CT_GAN_TEAM/CTGAN/.venv/bin/activate

pip install --upgrade pip
pip install tensorflow
pip install matplotlib pillow tensorflow_gan tensorflow_hub lpips

# # Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# python3 /blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/src/main.py --nE=1000 --iteration_path="Iteration_6/" --model_path="dcgan/" --disc_extra_steps=1
# python3 /blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/src/main.py --nE=5000 --iteration_path="Iteration_8/" --model_path="wgan/" --bs=64 --disc_extra_steps=5
# python3 /blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/src/main.py --nE=150 --iteration_path="Iteration_5/" --model_path="vae/" --bs=64
python3 /blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/src/main.py --nE=50 --iteration_path="Iteration_7/" --model_path="pggan/" --bs=16 --disc_extra_steps=1 --latent=512
