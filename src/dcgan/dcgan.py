import keras
import tensorflow as tf
from keras import layers
from abstract.model import GANModel

class DCGAN(GANModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.feature_maps = params.get('feature_maps', 64)

    def get_discriminator_model(self):
        feature_maps = self.feature_maps  # Use a local variable
        inputs = layers.Input(shape=self.image_size)
        x = layers.Conv2D(feature_maps, kernel_size=4, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.5)(x)

        # Progressively downsample
        for _ in range(4):
            feature_maps *= 2  # Update local variable
            x = layers.Conv2D(feature_maps, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Dropout(0.5)(x)
        
        x = layers.Flatten()(x)
        disc_output = layers.Dense(1, activation='sigmoid')(x)
        return keras.models.Model(inputs=inputs, outputs=disc_output, name="discriminator_fixed")

    def get_generator_model(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        fps = self.feature_maps
        # Use a Dense layer to map the input vector to a suitable number of units for reshaping
        x = layers.Dense(8*8*fps*8, use_bias=False)(inputs)
        x = layers.Reshape((8, 8, fps  * 8))(x)  # No change needed after this point
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Upsample to 16x16
        x = layers.Conv2DTranspose(fps * 8, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Upsample to 32x32
        x = layers.Conv2DTranspose(fps * 4, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Upsample to 64x64
        x = layers.Conv2DTranspose(fps * 2, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Upsample to 128x128
        x = layers.Conv2DTranspose(fps, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Upsample to 256x256
        output = layers.Conv2DTranspose(self.image_size[-1], kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')(x)
        
        return keras.models.Model(inputs=inputs, outputs=output, name="generator_fixed")
    
    def get_generator_optimizer(self):
       return tf.keras.optimizers.Adam(1e-4)

    def get_discriminator_optimizer(self):
       return tf.keras.optimizers.Adam(1e-4)
    
    def get_discriminator_loss(self, real_logits, fake_logits, real_images = None, fake_images = None):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
        fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
        total_loss = real_loss + fake_loss
        return total_loss
    
    # Define the loss functions for the generator.
    def get_generator_loss(self, fake_logits):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_logits), fake_logits)

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # Train the discriminator for d_steps iterations
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)

                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)

                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_loss = self.get_discriminator_loss(real_logits, fake_logits)

            # Get the gradients for the discriminator
            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)

            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)

            # Get the discriminator logits for fake images
            gen_logits = self.discriminator(generated_images, training=True)

            # Calculate the generator loss
            g_loss = self.get_generator_loss(gen_logits)

        # Get the gradients for the generator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)

        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss}
