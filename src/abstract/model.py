from abc import ABC, abstractmethod
import os
import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from evaluate.evaluate import ImageEvaluation

class AbstractModel(ABC):
    def __init__(self, params: dict):
        self.evaluate_only = params.get('evaluate_only', False)
        print("EVALUATE ONLY?", self.evaluate_only)
        self.epochs = params.get('nE', 100)
        self.save_interval = params.get('save_interval', 10)
        self.batch_size = params.get('bs', 32)
        self.latent_dim = params.get('latent', 128)
        self.num_img = params.get('num_img', 3)
        self.base_path = params.get('base_path', '../')
        self.original_iteration_path = params.get('iteration_path', 'Iteration_1/')
        self.model_path = params.get('model_path', 'Model/')
        self.covid_data_path = params.get('data_path', 'Data/')
        self.healthy_data_path = params.get('healthy_data_path', '/Healthy')
        self.image_size = params.get('image_size', (256, 256, 1))  
        self.evaluate_path = ""
        self.learning_rate = params.get('learning_rate', 0.0002)
        print("HEALTHY PATH", self.healthy_data_path)
        # Call preprocessing (this should be dataset-specific and implemented by subclasses)
        # self.train_data, self.test_data = self.preprocess_data()

    @abstractmethod
    def preprocess_data(self):
        """Preprocess data."""
        pass

    @abstractmethod
    def create_model(self):
        """Define the model architecture."""
        pass

    @abstractmethod
    def train_model(self):
        """Train the model."""
        pass

    @abstractmethod
    def get_final_images(self):
        """Get final images to train res net with."""
        pass

    def evaluate_model(self):
        """Evaluate the model."""
        result_path = self.base_path + "Outputs/evaluate/" + self.model_path + self.original_iteration_path + "evaluation.txt"
        healthy_image_path = self.base_path + self.healthy_data_path
        covid_image_path = self.base_path + self.covid_data_path
        artificial_image_path = self.base_path + "Outputs/evaluate/" + self.model_path + self.original_iteration_path
        evaluator = ImageEvaluation(healthy_image_path, covid_image_path, artificial_image_path, result_path, self.image_size)
        evaluator.run()

    def run(self):
        """Run the model."""
        if not self.evaluate_only:
            self.modality = "healthy"
            self.iteration_path = self.original_iteration_path + self.modality + "/"
            self.train_data, self.test_data = self.preprocess_data()
            print("EVALUATE IMAGES", self.evaluate_images_num, len(self.train_data))
            self.create_model()
            self.train_model()
            self.get_final_images()
            print("finished healthy")

            self.modality = "covid"
            self.iteration_path = self.original_iteration_path + self.modality + "/"
            self.train_data, self.test_data = self.preprocess_data()
            print("got covid data")
            self.create_model()
            print("created covid model")
            self.train_model()
            print("trained covid model")
            self.get_final_images()
            print("got covid final images")

        self.evaluate_model()

class GANModel(AbstractModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.d_steps = params.get('disc_extra_steps', 3)

    def preprocess_data(self):
        """GAN-specific data preprocessing."""
        base_path = self.base_path
        if self.modality == "covid":
            base_path += self.covid_data_path
        else:
            base_path += self.healthy_data_path
        
        print("PICTURE PATH", base_path)

        # List patient directories
        patient_dirs = sorted([d for d in os.listdir(base_path) if 'Patient' in d])
        print("Patient dirs", patient_dirs)

        train_dirs_index = (len(patient_dirs) * 3) // 4

        # Select first 60 for training and next 20 for testing
        train_dirs = patient_dirs[:train_dirs_index]
        test_dirs = patient_dirs[train_dirs_index:]

        print(train_dirs)
        print(test_dirs)

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
        avg_dimensions = self.image_size[:2]

        # Function to load images and convert to vectors
        def load_and_vectorize_images(image_paths, avg_dimensions):
            images = []
            for image_path in image_paths:
                img = Image.open(image_path)
                img = img.convert('L')
                img_resized = img.resize(avg_dimensions)
                img_vector = np.array(img_resized, dtype=np.float32)
                images.append(img_vector)
            return np.stack(images, axis=0)

        # Load and vectorize images
        train_images = load_and_vectorize_images(train_image_paths, avg_dimensions)
        test_images = load_and_vectorize_images(test_image_paths, avg_dimensions)
        
        train_images = train_images.reshape(train_images.shape[0], *self.image_size).astype("float32")
        train_images = (train_images - 127.5) / 127.5

        test_images = test_images.reshape(test_images.shape[0], *self.image_size).astype("float32")
        test_images = (test_images - 127.5) / 127.5

        self.steps_per_epoch = len(train_images) // self.batch_size
        self.steps = self.steps_per_epoch * self.epochs
        self.evaluate_images_num = len(train_images) // 2
        if len(train_images) == 0:
            raise ValueError("Training dataset is empty. Check your data paths and preprocessing.")

        return train_images, test_images

    def batch_generator(self, data, batch_size):
        num_samples = data.shape[0]
        indices = np.arange(num_samples)
        while True:
            np.random.shuffle(indices)
            for offset in range(0, num_samples, batch_size):
                if offset + batch_size <= num_samples:
                    batch_indices = indices[offset:offset + batch_size]
                    batch_images = data[batch_indices]
                    batch_images = self.augment_images(batch_images)
                    yield batch_images

    def augment_images(self, images):
        for i in range(images.shape[0]):
            if np.random.rand() < 0.5:
                images[i] = np.fliplr(images[i])
        return images
    
    def on_epoch_begin(self, epoch):
        self.current_epoch = epoch

    def on_batch_begin(self, step):
        pass
    
    def on_epoch_end(self, epoch):
        """Performs callback actions like saving images, models, and losses."""
        epoch+=1
        if epoch % self.save_interval == 0:
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5

            for i in range(self.num_img):
                image_path = self.base_path + "Outputs/images/" + self.model_path + self.iteration_path + str(i)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                img = generated_images[i].numpy()
                img = tf.keras.utils.array_to_img(img)
                img.save(image_path + "/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
            
            model_path = self.base_path + "Outputs/models/" + self.model_path + self.iteration_path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            self.discriminator.save(model_path + "discriminator_{epoch}.weights.h5".format(epoch=epoch))
            self.evaluate_path = model_path + "generator_{epoch}.weights.h5".format(epoch=epoch)
            self.generator.save(self.evaluate_path)

            # create a plot of the dloss and save it
            loss_path = self.base_path + "Outputs/losses/" + self.model_path + self.iteration_path
            if not os.path.exists(loss_path):
                os.makedirs(loss_path)
            plt.plot(self.dloss)
            plt.title('Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(loss_path + "dloss_{epoch}.png".format(epoch=epoch))
            plt.close()
            
            # create a plot of the gloss and save it
            plt.plot(self.gloss)
            plt.title('Generator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(loss_path + "gloss_{epoch}.png".format(epoch=epoch))

    def train_model(self):
        """Train the GAN-specific model."""

        # Create a batch generator for training
        train_gen = self.batch_generator(self.train_data, self.batch_size)

        # Calculate the number of steps per epoch
        steps_per_epoch = len(self.train_data) // self.batch_size

        for epoch in range(self.epochs):
            self.on_epoch_begin(epoch)
            for step in range(steps_per_epoch):
                # Get a batch of real images
                real_images = next(train_gen)

                # Train the discriminator and generator
                losses = self.train_step(real_images)

                # Track the losses for the current epoch
                self.dloss.append(losses['d_loss'])
                self.gloss.append(losses['g_loss'])

            # Perform the callback actions at the end of the epoch
            self.on_epoch_end(epoch)
            
            # Optional: print current losses for tracking
            print(f"Epoch {epoch+1}/{self.epochs}, d_loss: {losses['d_loss']}, g_loss: {losses['g_loss']}")
    
    def create_model(self):
        K.clear_session() 
        # tf.compat.v1.reset_default_graph()
        # tf.config.experimental.clear_devices()
        self.dloss = []
        self.gloss = []
        self.current_epoch = 0
        self.generator = self.get_generator_model()
        self.discriminator = self.get_discriminator_model()
        self.generator_optimizer = self.get_generator_optimizer()
        self.discriminator_optimizer = self.get_discriminator_optimizer()

    def get_final_images(self):
        K.clear_session()
        generator = load_model(self.evaluate_path)
        total_images = self.evaluate_images_num
        batch_size = 16  # Adjust this number based on your GPU capacity
        test_image_path = self.base_path + "Outputs/evaluate/" + self.model_path + self.iteration_path
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        for i in range(0, total_images, batch_size):
            current_batch_size = min(batch_size, total_images - i)
            random_noise = tf.random.normal(shape=(current_batch_size, self.latent_dim))
            generated_images = generator(random_noise)
            generated_images = (generated_images * 127.5) + 127.5
            for j in range(current_batch_size):
                idx = i + j
                img = tf.keras.utils.array_to_img(generated_images[j])
                img.save(test_image_path + f"generated_img_{idx}.png")

        
    @abstractmethod
    def get_discriminator_model(self):
        """Create discriminator model."""
        pass
    
    @abstractmethod
    def get_discriminator_loss(self, real_logits, fake_logits, real_images = None, fake_images = None):
        """Define discriminator loss."""
        pass

    @abstractmethod
    def get_generator_loss(self, fake_logits):
        """Define generator loss."""
        pass
    
    @abstractmethod
    def get_generator_optimizer(self):
        """Create generator optimizer."""
        pass

    @abstractmethod
    def get_discriminator_optimizer(self):
        """Create discriminator optimizer."""
        pass

    @abstractmethod
    def get_generator_model(self):
        """Create generator model."""
        pass

    @abstractmethod
    def train_step(self, real_images):
        """Train GAN-specific model."""
        pass  # Implement GAN-specific training loop
