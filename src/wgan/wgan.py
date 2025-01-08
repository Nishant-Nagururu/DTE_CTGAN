
import keras
import tensorflow as tf
from keras import layers
from abstract.model import GANModel


class WGAN(GANModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.d_steps = params.get('disc_extra_steps', 3)
        self.gp_weight = params.get('gp_weight', 10.0)
    
    def get_discriminator_model(self):
        def conv_block(
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            use_bn=False,
            use_dropout=False,
            drop_value=0.5,
        ):
            x = layers.Conv2D(
                filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
            )(x)
            if use_bn:
                x = layers.BatchNormalization()(x)
            x = activation(x)
            if use_dropout:
                x = layers.Dropout(drop_value)(x)
            return x

        img_input = layers.Input(shape=self.image_size)
        # Zero pad the input to make the input images size to (128, 128, 1).
        x = img_input
        x = conv_block(
            x,
            32,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=layers.LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        x = conv_block(
            x,
            64,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            use_bias=True,
            activation=layers.LeakyReLU(0.2),
            use_dropout=False,
            drop_value=0.3,
        )
        x = conv_block(
            x,
            128,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = conv_block(
            x,
            256,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=True,
            drop_value=0.3,
        )
        x = conv_block(
            x,
            512,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )
        x = conv_block(
            x,
            1024,
            kernel_size=(5, 5),
            strides=(2, 2),
            use_bn=False,
            activation=layers.LeakyReLU(0.2),
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )

        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)

        d_model = keras.models.Model(img_input, x, name="discriminator")
        return d_model

    def get_generator_model(self):
        def upsample_block(
            x,
            filters,
            activation,
            kernel_size=(3, 3),
            strides=(1, 1),
            up_size=(2, 2),
            padding="same",
            use_bn=False,
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        ):
            x = layers.UpSampling2D(up_size)(x)
            x = layers.Conv2D(
                filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
            )(x)

            if use_bn:
                x = layers.BatchNormalization()(x)

            if activation:
                x = activation(x)
            if use_dropout:
                x = layers.Dropout(drop_value)(x)
            return x
        
        noise = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(4 * 4 * 2048, use_bias=False)(noise)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Reshape((4, 4, 2048))(x)
        x = upsample_block(
            x,
            1024,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = upsample_block(
            x,
            512,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = upsample_block(
            x,
            256,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = upsample_block(
            x,
            128,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = upsample_block(
            x,
            64,
            layers.LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = upsample_block(
            x, 1, layers.Activation("tanh"), strides=(1, 1), up_size=(2,2), use_bias=False, use_bn=True
        )

        g_model = keras.models.Model(noise, x, name="generator")
        return g_model
    
    def get_generator_optimizer(self):
        generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9
        )
        return generator_optimizer

    def get_discriminator_optimizer(self):
        discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9
        )
        return discriminator_optimizer
    
    def get_discriminator_loss(self, real_logits, fake_logits, real_images = None, fake_images = None):
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def get_generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.get_discriminator_loss(real_logits, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight


            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.get_generator_loss(gen_img_logits)


        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}