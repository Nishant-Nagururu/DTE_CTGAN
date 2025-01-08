# CTGAN: A Collection of Generative Models for CT Image Synthesis

Welcome to **CTGAN**, a repository that provides multiple generative model implementations for CT image synthesis, including **DCGAN**, **WGAN**, **VAE**, and **PGGAN**. This codebase includes:

- Abstract classes that outline common steps (data preprocessing, model creation, training loops, and evaluation).
- Concrete model implementations: **DCGAN**, **WGAN**, **VAE**, and **PGGAN**.
- An **evaluation pipeline** that computes image similarity metrics (FID, LPIPS) and trains a ResNet-based classifier to gauge how well artificial images augment real data for classification tasks.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation & Requirements](#installation--requirements)
- [Running the Code](#running-the-code)
- [Key Arguments in `main.py`](#key-arguments-in-mainpy)
- [How It Works](#how-it-works)
- [Results & Evaluation](#results--evaluation)

---

## Project Structure

Once set up, the repository structure should look like this:


├── .venv                  # Contains installed packages
├── hpg_outputs            # Contains script output
│   ├── vasp.out           # Shows log output
│   ├── vasp.err           # Shows error output
├── Inputs
│   └── Covid              # Contains the COVID positive CT scans
│   └── Healthy            # Contains the healthy CT scans
├── Outputs                # For each model and each iteration of training said model, the code will create a unique folder within each of the four below folders
│   └── evaluate           # Contains the images produced by the evaluation script and an evaluation.txt with the final evaluation output
│   │   └── ...
│   └── images             # Contains images produced during training at set intervals
│   │   └── ...
│   └── losses             # Contains loss graphs produced during training at set intervals
│   │   └── ...
│   └── models             # Contains checkpoints produced during training at set intervals
│   │   └── ...
├── src/
│   ├── main.py            # Entry point script that parses arguments and runs the training/evaluation
│   ├── run.sh             # Slurm batch script for running the job on an HPC cluster
│   ├── abstract/
│   │   └── model.py       # Abstract classes implementation
│   ├── wgan/
│   │   └── wgan.py        # WGAN implementation
│   ├── vae/
│   │   └── vae.py         # VAE implementation
│   ├── dcgan/
│   │   └── dcgan.py       # DCGAN implementation
│   ├── pggan/
│   │   └── pggan.py       # PGGAN implementation
│   ├── evaluate/
│   │   └── evaluate.py    # ImageEvaluation class for computing FID, LPIPS, and training a classification model
├── README.md              # This README file


---

## Features

1. **Abstract Model Class**  
   A shared abstract base class outlines the primary functions (preprocessing, model creation, training, etc.) for all GAN models.

2. **Multiple GAN Implementations**  
   - **DCGAN**: A classical deep convolutional GAN model.
   - **WGAN**: A Wasserstein GAN variant with gradient penalty (`gp_weight`) for more stable training.
   - **VAE**: A Variational Autoencoder that uses a KL-divergence term for probabilistic generation.
   - **PGGAN**: A Progressive Growing GAN, which grows layers and resolution during training.

3. **Comprehensive Evaluation Pipeline**  
   - **FID (Frechet Inception Distance)**
   - **LPIPS (Learned Perceptual Image Patch Similarity)**
   - **Classifier-based evaluation** via a ResNet50 to check how generated images can augment real data.

4. **Easy Parameter Customization**  
   Adjust epochs, batch size, latent dimension size, image paths, etc., using command-line flags.

5. **Slurm Script for HPC**  
   A preconfigured Slurm script (`run.sh`) allows easy deployment on HPC clusters.

---

## Installation & Requirements

- **Python 3.8+**  
- **TensorFlow 2.7.0**  
- **PyTorch** (required for LPIPS calculation)  
- **NumPy, Matplotlib, Pillow**  
- **TensorFlow-GAN, TensorFlow-Hub** (for Inception-based FID)  

If you are running on an HPC, make sure to load the required modules (CUDA, Python, etc.). The provided `run.shh` Slurm script handles the environment setup on many HPC systems (e.g., UF HiPerGator).

---

## Running the Code

1. **Edit `run.sh`** to modify your job name, account, email, etc., if needed.
2. In the root directory, submit the job:
   ```bash
   sbatch run.sh
   ```
3. The script will set up a Python virtual environment, install packages, and run `main.py`. You may customize which model or models to train and with what parameters by editing run.sh.

---

## Key Arguments in `main.py`

Below are some essential flags you can pass:

| Argument             | Default                                         | Description                                                          |
|----------------------|-------------------------------------------------|----------------------------------------------------------------------|
| `--nE`              | `100`                                           | Number of training epochs                                            |
| `--save_interval`   | `10`                                            | Save model checkpoints and generated images every `save_interval` epochs |
| `--bs`              | `32`                                            | Batch size                                                           |
| `--latent`          | `128`                                           | Latent dimension size for the generator                              |
| `--num_img`         | `3`                                             | Number of images to generate during each callback                    |
| `--base_path`       | `/blue/dream_team/CT_GAN_TEAM/DTE_CTGAN/`       | Base directory containing data and outputs                           |
| `--iteration_path`  | `Iteration_1/`                                  | Subfolder path for iteration outputs                                 |
| `--model_path`      | `wgan/`                                         | Chooses which model to use (`dcgan/`, `wgan/`, `vae/`, `pggan/`)     |
| `--data_path`       | `Inputs/Covid/`                                 | Path to the COVID dataset                                            |
| `--healthy_data_path`| `Inputs/Healthy/`                              | Path to the Healthy dataset                                          |
| `--image_size`      | `256`                                           | Height/width of the input images (converted internally to `(256,256,channels)`) |
| `--image_channels`  | `1`                                             | Number of image channels (1 for grayscale, 3 for RGB)                |
| `--disc_extra_steps`| `3`                                             | Extra discriminator steps per training iteration (used in WGAN)      |
| `--gp_weight`       | `10.0`                                          | Gradient penalty weight (WGAN)                                       |
| `--capacity`        | `32`                                            | Capacity for the VAE                                                 |
| `--variational_beta`| `1.0`                                           | Beta parameter for the VAE                                          |
| `--learning_rate`   | `0.0002`                                        | Learning rate                                                        |
| `--evaluate_only`   | `False`                                         | Set to `True` to skip training and run only the evaluation           |

---

## How It Works

1. **Argument Parsing**:  
   `parse_arguments()` collects user-specified parameters (epochs, batch size, etc.).

2. **Model Initialization**:  
   Depending on `--model_path`, the code instantiates **DCGAN**, **WGAN**, **VAE**, or **PGGAN** classes.

3. **Training**:  
   - Data is split into **Healthy** and **COVID** subsets.  
   - The model trains on **Healthy** images first, then **COVID**.  
   - Each epoch can trigger a callback that saves model weights, generates sample images, and logs losses.

4. **Evaluation**:  
   - **FID** and **LPIPS** measure similarity between real and generated images.  
   - A **ResNet50** classifier is optionally trained with or without generated images to see if synthetic data improves classification performance.

---

## Results & Evaluation

- During training, generated samples and model checkpoints will be saved in:
  ```
  ./Outputs
  ├── images/
  ├── models/
  └── losses/
  ```
- Evaluation logs are stored in:
  ```
  ./evaluate/
  ```
- Metrics such as **FID** and **LPIPS** and classification accuracy are printed and recorded in `evaluation.txt`.

---


Thank you for using **CTGAN**! If you find this work helpful, please consider giving this repository a ⭐. If you have any issues or questions, feel free to contact me at nishant.nagururu@ufl.edu
