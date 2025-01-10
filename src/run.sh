#!/bin/bash
#SBATCH --job-name=DTE_CTGAN
#SBATCH --output=/blue/dream_team/CT_GAN_TEAM/DTE_CTGAN/hpg_outputs/vasp.out
#SBATCH --error=/blue/dream_team/CT_GAN_TEAM/DTE_CTGAN/hpg_outputs/vasp.err
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
#SBATCH --mem-per-gpu=255000
#SBATCH --time=72:00:00

# Change vasp paths above and change to your working directory below
cd /blue/dream_team/CT_GAN_TEAM/DTE_CTGAN 

# Unload all modules
module purge

# Load required modules
module load cuda/11.4.3 nvhpc/23.7 openmpi/4.1.5 vasp/6.4.1
module load python/3.8

# Set the XLA_FLAGS environment variable (if needed)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/apps/compilers/nvhpc/23.7/Linux_x86_64/23.7/cuda"

# Create and activate virtual environment using Python 3.8
python3.8 -m venv .venv
source /blue/dream_team/CT_GAN_TEAM/DTE_CTGAN/.venv/bin/activate

# Upgrade pip
pip install --upgrade pip
pip uninstall -y tensorflow tensorflow-estimator
# Install required packages with specific versions
pip install torch torchvision
pip install tensorflow
pip install tensorflow-estimator
pip install matplotlib pillow nbconvert jupyter tensorflow_gan tensorflow_hub lpips

# Verify the installation
pip list
which python
python3.8 -c "import tensorflow as tf; print(tf.__version__)"
python3.8 -c "import tensorflow as tf; print(tf.estimator)"

# Verify the installation:
python3.8 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Available GPUs:', tf.config.list_physical_devices('GPU'))"

cd src

# Choose which models to run and customize as needed
python3.8 main.py --nE=50 --iteration_path="Iteration_14/" --model_path="pggan/" --bs=8 --disc_extra_steps=1 --latent=512
python3.8 main.py --nE=1000 --iteration_path="Iteration_12/" --model_path="dcgan/" --disc_extra_steps=1
python3.8 main.py --nE=1500 --iteration_path="Iteration_12/" --model_path="wgan/" --bs=64 --disc_extra_steps=5 
python3.8 main.py --nE=150 --iteration_path="Iteration_12/" --model_path="vae/" --bs=64
