import keras
import tensorflow as tf
from keras import layers
from abstract.model import AbstractModel
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

"""
This code was adapted from the VAE implementation found on Kaggle:
https://www.kaggle.com/code/susanta21/vae-implementation

Original authors: Susanta Baidya
Date accessed: 01-14-2025

If you use this code in your research or projects, please consider citing the source:
@misc{kaggle_vae,
  author = {Susanta Baidya},
  title = {VAE Implementation},
  howpublished = {\url{https://www.kaggle.com/code/susanta21/vae-implementation}},
  note = {Accessed: 2025-01-14}
}
"""

@tf.keras.utils.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(AbstractModel):
    def __init__(self, params: dict):
        self.capacity = params.get('capacity', 32)
        self.weight_decay = params.get('weight_decay', 1e-4)
        self.variational_beta = params.get('variational_beta', 1.0)
        self.loss = []
        super().__init__(params)
    
    def preprocess_data(self):
        """GAN-specific data preprocessing with modality consideration."""
        train_image_paths, test_image_paths = self.get_test_and_train_paths()

        avg_dimensions = self.image_size[:2]

        # Load and vectorize images
        train_images = self.load_and_vectorize_images(train_image_paths, avg_dimensions)
        test_images = self.load_and_vectorize_images(test_image_paths, avg_dimensions)

        train_images = train_images.reshape(train_images.shape[0], *self.image_size).astype("float32")
        train_images = train_images / 255.0

        test_images = test_images.reshape(test_images.shape[0], *self.image_size).astype("float32")
        test_images = test_images / 255.0

        self.evaluate_images_num = len(train_images)//2
        return (
            tf.data.Dataset.from_tensor_slices(train_images)
            .shuffle(len(train_images))
            .batch(self.batch_size),
            test_images,
        )


    def build_encoder(self):
        inputs = tf.keras.layers.Input(shape=self.image_size)
        
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        
        # Outputs for mean and log variance
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        
        encoder = tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')
        return encoder

    # Decoder
    def build_decoder(self):
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))
        
        x = tf.keras.layers.Dense(16 * 16 * 256, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((16, 16, 256))(x)
        x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        outputs = tf.keras.layers.Conv2DTranspose(1, 3, strides=1, padding='same')(x)
        
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
        return decoder

    def create_model(self):
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        
        # Define the VAE model
        inputs = tf.keras.layers.Input(shape=self.image_size)
        
        # Encode input to latent space
        z_mean, z_log_var = encoder(inputs)
        
        # Apply reparameterization trick using the custom Sampling layer
        z = Sampling()([z_mean, z_log_var])
        
        # Decode back to original image
        reconstructed = decoder(z)
        
        # Define the complete VAE model
        vae = tf.keras.Model(inputs, reconstructed, name='vae')
        self.vae = vae
        
        # Initialize optimizer
        self.optimizer = self.get_vae_optimizer()

    def get_vae_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    def compute_loss(self, x):
        reconstructed = self.vae(x)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, reconstructed)) * self.image_size[0] * self.image_size[1]
        
        z_mean, z_log_var = self.vae.get_layer('encoder')(x)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        
        return reconstruction_loss + kl_loss
    
    # @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.vae.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_variables))
        return loss
    
    def save_and_generate_images(self, epoch):
        # Generate 3 images from random latent vectors
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))  # Latent space sampling

        # Use the decoder part of the VAE to generate images
        generated_images = self.vae.get_layer('decoder')(random_latent_vectors)

        for i in range(self.num_img):
            image_path = self.base_path + "Outputs/images/" + self.model_path + self.iteration_path + f"{i}"
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            img = tf.keras.utils.array_to_img(generated_images[i])
            img.save(image_path + f"/generated_img_{i}_{epoch}.png")

        # Save model in TensorFlow's SavedModel format
        model_path = self.base_path + "Outputs/models/" + self.model_path + self.iteration_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.evaluate_path = model_path + f"vae_{epoch}.weights.h5"
        self.vae.save(self.evaluate_path)

        # Save loss plot
        loss_path = self.base_path + "Outputs/losses/" + self.model_path + self.iteration_path
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)

        # Plot the loss history
        plt.plot(self.loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Loss')
        plt.legend()
        plt.savefig(loss_path + f"loss_{epoch}.png")
        plt.close()


    def train_model(self):
        for epoch in range(1, self.epochs + 1):
            for train_x in self.train_data:
                loss = self.train_step(train_x)
            
            # Track loss
            self.loss.append(loss.numpy())
            
            if epoch % self.save_interval == 0:
                self.save_and_generate_images(epoch)
                print(f"Epoch {epoch}: Loss {loss.numpy()}")
    
    def get_final_images(self):
        vae = load_model(self.evaluate_path)
        total_images = self.evaluate_images_num
        batch_size = 16  # Adjust this number based on your GPU capacity
        test_image_path = self.base_path + "Outputs/evaluate/" + self.model_path + self.iteration_path
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        for i in range(0, total_images, batch_size):
            current_batch_size = min(batch_size, total_images - i)
            random_noise = tf.random.normal(shape=(current_batch_size, self.latent_dim))
            generated_images = vae.get_layer('decoder')(random_noise)
            generated_images = (generated_images * 127.5) + 127.5
            for j in range(current_batch_size):
                idx = i + j
                img = tf.keras.utils.array_to_img(generated_images[j])
                img.save(test_image_path + f"generated_img_{idx}.png")

