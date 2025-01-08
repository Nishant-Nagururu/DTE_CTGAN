import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from PIL import Image
import os
from abstract.model import GANModel

# @tf.keras.utils.register_keras_serializable()
# class PixelNormalization(Layer):
#     def __init__(self, **kwargs):
#         super(PixelNormalization, self).__init__(**kwargs)

#     def call(self, inputs):
#         mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
#         l2 = tf.math.rsqrt(mean_square + 1.0e-8)
#         normalized = inputs * l2
#         return normalized

#     def compute_output_shape(self, input_shape):
#         return input_shape

# # Calculate the average standard deviation of all features and spatial location.
# # Concat after creating a constant feature map with the average standard deviation
# @tf.keras.utils.register_keras_serializable()
# class MinibatchStdev(Layer):
#     def __init__(self, **kwargs):
#         super(MinibatchStdev, self).__init__(**kwargs)
    
#     def call(self, inputs):
#         mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
#         stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
#         average_stddev = tf.reduce_mean(stddev, keepdims=True)
#         shape = tf.shape(inputs)
#         minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
#         combined = tf.concat([inputs, minibatch_stddev], axis=-1)
        
#         return combined
    
#     def compute_output_shape(self, input_shape):
#         input_shape = list(input_shape)
#         input_shape[-1] += 1
#         return tuple(input_shape)

# # Perform Weighted Sum
# # Define alpha as backend.variable to update during training
# @tf.keras.utils.register_keras_serializable()
# class WeightedSum(Add):
#     def __init__(self, alpha=0.0, **kwargs):
#         super(WeightedSum, self).__init__(**kwargs)
#         self.alpha = backend.variable(alpha, name='ws_alpha')
    
#     def _merge_function(self, inputs):
#         assert (len(inputs) == 2)
#         output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
#         return output

# # Scale by the number of input parameters to be similar dynamic range  
# # For details, refer to https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
# # stddev = sqrt(2 / fan_in)
# @tf.keras.utils.register_keras_serializable()
# class WeightScaling(Layer):
#     def __init__(self, shape, gain=np.sqrt(2), **kwargs):
#         super(WeightScaling, self).__init__(**kwargs)
#         shape = np.asarray(shape)
#         shape = tf.constant(shape, dtype=tf.float32)
#         fan_in = tf.math.reduce_prod(shape)
#         self.wscale = gain * tf.math.rsqrt(fan_in)

#     def call(self, inputs, **kwargs):
#         inputs = tf.cast(inputs, tf.float32)
#         return inputs * self.wscale

#     def compute_output_shape(self, input_shape):
#         return input_shape

# def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
#     init = RandomNormal(mean=0., stddev=1.)
#     in_filters = backend.int_shape(x)[-1]
#     x = layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
#     x = WeightScaling(shape=(in_filters), gain=gain)(x)
#     x = Bias(input_shape=x.shape)(x)
#     if activate=='LeakyReLU':
#         x = layers.LeakyReLU(0.2)(x)
#     elif activate=='tanh':
#         x = layers.Activation('tanh')(x)
    
#     if use_pixelnorm:
#         x = PixelNormalization()(x)
#     return x

# def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1)):
#     init = RandomNormal(mean=0., stddev=1.)
#     in_filters = backend.int_shape(x)[-1]
#     x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
#     x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
#     x = Bias(input_shape=x.shape)(x)
#     if activate=='LeakyReLU':
#         x = layers.LeakyReLU(0.2)(x)
#     elif activate=='tanh':
#         x = layers.Activation('tanh')(x)
    
#     if use_pixelnorm:
#         x = PixelNormalization()(x)
#     return x 

# @tf.keras.utils.register_keras_serializable()
# class Bias(Layer):
#     def __init__(self, **kwargs):
#         super(Bias, self).__init__(**kwargs)

#     def build(self, input_shape):
#         b_init = tf.zeros_initializer()
#         self.bias = tf.Variable(initial_value = b_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)  

#     def call(self, inputs, **kwargs):
#         return inputs + self.bias
    
#     def compute_output_shape(self, input_shape):
#         return input_shape  

@tf.keras.utils.register_keras_serializable()
class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        normalized = inputs * l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(PixelNormalization, self).get_config()
        # No additional arguments to add
        return config

@tf.keras.utils.register_keras_serializable()
class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
    
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)
        return combined
    
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)
    
    def get_config(self):
        config = super(MinibatchStdev, self).get_config()
        # No additional arguments to add
        return config

@tf.keras.utils.register_keras_serializable()
class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')
    
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
        return output

    def get_config(self):
        config = super(WeightedSum, self).get_config()
        # Serialize the alpha value
        config.update({
            'alpha': float(self.alpha.numpy()),
        })
        return config

@tf.keras.utils.register_keras_serializable()
class WeightScaling(Layer):
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        self.original_shape = shape  # Store the original shape
        self.gain = gain
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain * tf.math.rsqrt(fan_in)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(WeightScaling, self).get_config()
        # Serialize the shape and gain
        config.update({
            'shape': self.original_shape,
            'gain': self.gain,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.bias = self.add_weight(
            shape=(input_shape[-1],),
            initializer=b_init,
            trainable=True,
            name='bias',
            dtype='float32'
        )

    def call(self, inputs, **kwargs):
        return inputs + self.bias
    
    def compute_output_shape(self, input_shape):
        return input_shape  

    def get_config(self):
        config = super(Bias, self).get_config()
        # No additional arguments to add
        return config

def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = tf.keras.layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters,), gain=gain)(x)
    x = Bias()(x)
    if activate == 'LeakyReLU':
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = tf.keras.layers.Activation('tanh')(x)
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x

def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias()(x)
    if activate == 'LeakyReLU':
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = tf.keras.layers.Activation('tanh')(x)
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x 

class PGGAN(GANModel):
    def __init__(self, params: dict):
        self.filters = params.get('filters', [512, 512, 512, 512, 256, 128, 64, 32])
        self.drift_weight = params.get('drift_weight', 0.001)
        self.gp_weight = params.get('gp_weight', 10.0)
        self.n_depth = 0
        super().__init__(params)
    
    def create_model(self):
        self.n_depth = 0
        self.prefix = '0_init'
        super().create_model()

    def preprocess_data(self):
        """GAN-specific data preprocessing."""
        base_path = self.base_path + self.covid_data_path

        # List patient directories
        patient_dirs = sorted([d for d in os.listdir(base_path) if 'Patient' in d])

        train_dirs_index = (len(patient_dirs) * 3) // 4

        train_dirs = patient_dirs[:train_dirs_index]
        test_dirs = patient_dirs[train_dirs_index:]

        # List all image paths
        train_image_paths = [os.path.join(base_path, patient, image) 
                            for patient in train_dirs 
                            for image in os.listdir(os.path.join(base_path, patient))]

        test_image_paths = [os.path.join(base_path, patient, image) 
                            for patient in test_dirs 
                            for image in os.listdir(os.path.join(base_path, patient))]

        # Remove any paths that do not end in .png
        train_image_paths = [path for path in train_image_paths if path.endswith('.png')]
        test_image_paths = [path for path in test_image_paths if path.endswith('.png')]
        curr_dimensions = (4*(2**self.n_depth), 4*(2**self.n_depth))

        # Function to load images and convert to vectors
        def load_and_vectorize_images(image_paths, avg_dimensions):
            images = []
            for image_path in image_paths:
                img = Image.open(image_path)
                img = img.convert('RGB')
                img_resized = img.resize(avg_dimensions)
                img_vector = np.array(img_resized, dtype=np.float32)
                images.append(img_vector)
            return np.stack(images, axis=0)

        # Load and vectorize images
        train_images = load_and_vectorize_images(train_image_paths, curr_dimensions)
        test_images = load_and_vectorize_images(test_image_paths, curr_dimensions)
        
        curr_dimensions = (4*(2**self.n_depth), 4*(2**self.n_depth), 3)
        train_images = train_images.reshape(train_images.shape[0], *curr_dimensions).astype("float32")
        train_images = (train_images - 127.5) / 127.5
        # train_images = train_images / 255.0

        test_images = test_images.reshape(test_images.shape[0], *curr_dimensions).astype("float32")
        test_images = (test_images - 127.5) / 127.5
        # test_images = test_images / 255.0

        self.steps_per_epoch = len(train_images) // self.batch_size
        self.steps = self.steps_per_epoch * self.epochs
        self.evaluate_images_num = len(train_images) // 2

        return train_images, test_images
    
    def compile(self):
        self.discriminator_optimizer = self.get_discriminator_optimizer()
        self.generator_optimizer = self.get_generator_optimizer()
    
    def get_generator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    
    def get_discriminator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    
    def get_generator_model(self):
        noise = layers.Input(shape=(self.latent_dim,))
        x = PixelNormalization()(noise)
        # Actual size(After doing reshape) is just FILTERS[0], so divide gain by 4
        x = WeightScalingDense(x, filters=4*4*self.filters[0], gain=np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=True)
        x = layers.Reshape((4, 4, self.filters[0]))(x)

        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)

        # Add "toRGB", the original paper uses linear as actiavation. 
        # Gain should be 1, cos it's a last layer 
        x = WeightScalingConv(x, filters=3, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False)

        g_model = Model(noise, x, name='generator')
        g_model.summary()
        return g_model

    def fade_in_generator(self):
        #for layer in self.generator.layers:
        #    layer.trainable = False
        # 1. Get the node above the “toRGB” block 
        block_end = self.generator.layers[-5].output
        # 2. Double block_end       
        block_end = layers.UpSampling2D((2,2))(block_end)

        # 3. Reuse the existing “toRGB” block defined as“x1”. 
        x1 = self.generator.layers[-4](block_end) # Conv2d
        x1 = self.generator.layers[-3](x1) # WeightScalingLayer
        x1 = self.generator.layers[-2](x1) # Bias
        x1 = self.generator.layers[-1](x1) #tanh

        # 4. Define a "fade in" block (x2) with two 3x3 convolutions and a new "toRGB".
        x2 = WeightScalingConv(block_end, filters=self.filters[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        
        x2 = WeightScalingConv(x2, filters=3, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False)

        # Define stabilized(c. state) generator
        self.generator_stabilize = Model(self.generator.input, x2, name='generator')

        # 5.Then "WeightedSum" x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator')

        self.generator.summary()

    # Change to stabilized(c. state) generator 
    def stabilize_generator(self):
        self.generator = self.generator_stabilize

        self.generator.summary()

    def get_discriminator_model(self):
        img_input = layers.Input(shape = (4,4,3))
        # img_input = tf.cast(img_input, tf.float32)
        img_input = layers.Lambda(lambda x: tf.cast(x, tf.float32))(img_input)
        
        # fromRGB
        x = WeightScalingConv(img_input, filters=self.filters[0], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU')
        
        # Add Minibatch end of discriminator
        x = MinibatchStdev()(x)

        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')
        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4,4))

        x = layers.Flatten()(x)
        # Gain should be 1, cos it's a last layer 
        x = WeightScalingDense(x, filters=1, gain=1.)

        d_model = Model(img_input, x, name='discriminator')

        return d_model
    
    def fade_in_discriminator(self):
        #for layer in self.discriminator.layers:
        #    layer.trainable = False
        input_shape = list(self.discriminator.input.shape)
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])
        img_input = layers.Input(shape = input_shape)
        # img_input = tf.cast(img_input, tf.float32)
        img_input = layers.Lambda(lambda x: tf.cast(x, tf.float32))(img_input)

        # 2. Add pooling layer 
        #    Reuse the existing “formRGB” block defined as “x1".
        x1 = layers.AveragePooling2D(pool_size=(2, 2))(img_input)
        x1 = self.discriminator.layers[1](x1) # Conv2D FromRGB
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer
        x1 = self.discriminator.layers[3](x1) # Bias
        x1 = self.discriminator.layers[4](x1) # LeakyReLU

        # 3.  Define a "fade in" block (x2) with a new "fromRGB" and two 3x3 convolutions. 
        #     Add an AveragePooling2D layer
        x2 = WeightScalingConv(img_input, filters=self.filters[self.n_depth], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = layers.AveragePooling2D(pool_size=(2, 2))(x2)

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(5, len(self.discriminator.layers)):
            x2 = self.discriminator.layers[i](x2)
        self.discriminator_stabilize = Model(img_input, x2, name='discriminator')

        # 5. Add existing discriminator layers. 
        for i in range(5, len(self.discriminator.layers)):
            x = self.discriminator.layers[i](x)
        self.discriminator = Model(img_input, x, name='discriminator')

        self.discriminator.summary()

    # Change to stabilized(c. state) discriminator 
    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize
        self.discriminator.summary()

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def get_discriminator_loss(self, real_logits, fake_logits, real_images = None, fake_images = None):
        d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

        # Calculate the gradient penalty
        gp = self.gradient_penalty(self.batch_size, real_images, fake_images)

        # Calculate the drift for regularization
        drift = tf.reduce_mean(tf.square(real_logits))

        # Add the gradient penalty to the original discriminator loss
        d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift

        return d_loss
    
    def get_generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)

        
    # def get_discriminator_loss(self, real_logits, fake_logits, real_images = None, fake_images = None):
    #     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #     real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    #     fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    #     total_loss = real_loss + fake_loss
    #     return total_loss
    
    # # Define the loss functions for the generator.
    # def get_generator_loss(self, fake_logits):
    #     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #     return cross_entropy(tf.ones_like(fake_logits), fake_logits)


    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                d_loss = self.get_discriminator_loss(real_logits, fake_logits, real_images, fake_images)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer

            self.discriminator_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            # g_loss = -tf.reduce_mean(gen_img_logits)
            g_loss = self.get_generator_loss(gen_img_logits)
        # Get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
        return {'d_loss': d_loss, 'g_loss': g_loss}
   
    def on_epoch_end(self, epoch):
        """Performs callback actions like saving images, models, and losses."""
        epoch+=1
        if epoch % self.save_interval == 0:
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5

            for i in range(self.num_img):
                image_path = self.base_path + "Outputs/images/" + self.model_path + self.iteration_path + str(i) + "/"
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                img = generated_images[i].numpy()
                img = tf.keras.utils.array_to_img(img)
                img.save(image_path + f"{self.prefix}_img_{i}_{epoch}.png")
            
            model_path = self.base_path + "Outputs/models/" + self.model_path + self.iteration_path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            self.discriminator.save(model_path + f"discriminator_{self.prefix}_{epoch}.weights.h5")
            self.evaluate_path = model_path + f"generator_{self.prefix}_{epoch}.weights.h5"
            self.generator.save(self.evaluate_path)

            # create a plot of the dloss and save it
            loss_path = self.base_path + "Outputs/losses/" + self.model_path + self.iteration_path
            if not os.path.exists(loss_path):
                os.makedirs(loss_path)
            plt.plot(self.dloss)
            plt.title('Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(loss_path + f"dloss_{epoch}.png")
            plt.close()
            
            # create a plot of the gloss and save it
            plt.plot(self.gloss)
            plt.title('Generator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(loss_path + f"gloss_{epoch}.png")


    def on_batch_begin(self, batch, logs=None):
        # Update alpha in WeightedSum layers
        alpha = ((self.current_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1)
        #print(f'\n {self.steps}, {self.n_epoch}, {self.steps_per_epoch}, {alpha}')
        for layer in self.generator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)
        for layer in self.discriminator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


    def train_model(self):
        """Train the GAN-specific model."""

        for n_depth in range(1, 7):
            self.compile()

            self.n_depth = n_depth

            train_images, _ = self.preprocess_data()  # Get the training images

            # Create a batch generator for training
            train_gen = self.batch_generator(train_images, self.batch_size)

            self.prefix = f'{n_depth}_fade_in'

            self.fade_in_discriminator()
            self.fade_in_generator()

            self.compile()

            for epoch in range(self.epochs):
                self.on_epoch_begin(epoch)
                step_count = 0
                for step in range(self.steps_per_epoch):
                    step_count += 1
                    self.on_batch_begin(step)
                    # Get a batch of real images
                    real_images = next(train_gen)
                    if len(real_images) < self.batch_size:
                        break

                    # Train the discriminator and generator
                    losses = self.train_step(real_images)

                    # Track the losses for the current epoch
                    self.dloss.append(losses['d_loss'])
                    self.gloss.append(losses['g_loss'])

                    if step_count == 100:
                        break

                # Perform the callback actions at the end of the epoch
                self.on_epoch_end(epoch)
                
                # Optional: print current losses for tracking
                print(f"Epoch {epoch+1}/{self.epochs}, d_loss: {losses['d_loss']}, g_loss: {losses['g_loss']}")

            fade_in_model_path = self.base_path + "Outputs/models/" + self.model_path + self.iteration_path + f'pgan_{self.prefix}.weights.h5'
            self.generator.save(fade_in_model_path)

            # Switch to stabilized generator and discriminator for the next phase
            self.prefix = f'{n_depth}_stabilize'

            self.stabilize_generator()
            self.stabilize_discriminator()

            self.compile()

            for epoch in range(self.epochs):
                print(f"\nStarting epoch {epoch + 1}/{self.epochs} at depth {n_depth} (stabilize)")
                self.on_epoch_begin(epoch)

                for step in range(self.steps_per_epoch):
                    self.on_batch_begin(step)

                    # Get a batch of real images
                    real_images = next(train_gen)

                    # Perform the training step
                    losses = self.train_step(real_images)

                    # Track the losses
                    self.dloss.append(losses['d_loss'])
                    self.gloss.append(losses['g_loss'])
                
                    # Optionally log progress
                    print(f"Epoch {epoch + 1}/{self.epochs}, Step {step + 1}/{self.steps_per_epoch}, "
                        f"d_loss: {losses['d_loss']:.4f}, g_loss: {losses['g_loss']:.4f}")

                self.on_epoch_end(epoch)  # End of stabilization epoch

            # Save weights after stabilization phase
            stabilize_model_path = self.base_path + "Outputs/models/" + self.model_path + self.iteration_path + f'pgan_{self.prefix}.weights.h5'
            self.generator.save(stabilize_model_path)