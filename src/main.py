import argparse
import os
from wgan.wgan import WGAN
from vae.vae import VAE
from dcgan.dcgan import DCGAN
from pggan.pggan import PGGAN


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a GAN model with given parameters.')

    # Add all the command line arguments you want to accept
    parser.add_argument('--nE', type=int, default=100, help='Number of epochs to train the GAN model')
    parser.add_argument('--save_interval', type=int, default = 10, help='Number of epochs in between each time the model is saved.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size for training')
    parser.add_argument('--latent', type=int, default=128, help='Latent dimension size for the generator')
    parser.add_argument('--num_img', type=int, default=3, help='Number of images to generate during callback')
    parser.add_argument('--base_path', type=str, default='/blue/dream_team/CT_GAN_TEAM/CTGAN_FINAL_COPY/', help='Base path for the data and output folders')
    parser.add_argument('--iteration_path', type=str, default='Iteration_1/', help='Path to save iteration outputs')
    parser.add_argument('--model_path', type=str, default='wgan/', help='Path to save model checkpoints')
    parser.add_argument('--data_path', type=str, default='Inputs/New_Data_CoV2/Covid/', help='Path to the covid dataset')
    parser.add_argument('--healthy_data_path', type=str, default='Inputs/New_Data_CoV2/Healthy', help='Path to healthy data')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (height, width)')
    parser.add_argument('--image_channels', type=int, default=1, help='Image channels (1 or 3)')
    parser.add_argument('--feature_maps', type=int, default=64, help='Number of feature maps in the first layer of the generator for DCGAN')
    parser.add_argument('--disc_extra_steps', type=int, default=3, help='Number of extra steps for the discriminator in WGAN')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Weight for the gradient penalty in WGAN')
    parser.add_argument('--capacity', type=int, default=32, help='Capacity of the VAE model')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the VAE model')
    parser.add_argument('--variational_beta', type=float, default=1.0, help='Variational beta for the VAE model')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning Rate')
    parser.add_argument('--evaluate_only', type=bool, default=False, help='Set to true if you only want evaluate and not train processes')
    
    return parser.parse_args()

def main():
    # Parse the arguments
    args = parse_arguments()

    # Convert arguments to a dictionary
    params = {
        'nE': args.nE,
        'save_interval': args.save_interval,
        'bs': args.bs,
        'latent': args.latent,
        'num_img': args.num_img,
        'base_path': args.base_path,
        'iteration_path': args.iteration_path,
        'model_path': args.model_path,
        'data_path': args.data_path,
        'healthy_data_path': args.healthy_data_path,
        'image_size': tuple([args.image_size, args.image_size, args.image_channels]),
        'feature_maps': args.feature_maps,
        'disc_extra_steps': args.disc_extra_steps,
        'gp_weight': args.gp_weight,
        'capacity': args.capacity,
        'weight_decay': args.weight_decay,
        'variational_beta': args.variational_beta,
        'learning_rate': args.learning_rate,
        'evaluate_only': args.evaluate_only
    }

    # Initialize the GAN model with the provided parameters
    # gan_model = DCGAN(params)
    model = None
    if args.model_path == 'dcgan/':
        model = DCGAN(params)
    elif args.model_path == 'vae/':
        model = VAE(params)
    elif args.model_path == 'wgan/':
        model = WGAN(params)
    elif args.model_path == 'pggan/':
        model = PGGAN(params)
    else:
        raise ValueError('Invalid model path. Please choose one of the following: dcgan/, vae/, wgan/')

    # Train the GAN model
    # model.train_model()
    model.run()

if __name__ == "__main__":
    main()
