import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import random
from PIL import Image

import tensorflow_gan as tfgan
import tensorflow_hub as hub
import lpips
import torch
# from tensorflow.keras.mixed_precision import set_global_policy

class ImageEvaluation:  
    def set_seed(self, seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def __init__(self, healthy_path, covid_path, artificial_path, results_path, image_size):
        # Commented out mixed precision policy for now
        # set_global_policy('mixed_float16')

        # Ensure image_size has 3 channels
        self.set_seed()
        self.image_size = list(image_size)
        if self.image_size[-1] != 3:
            self.image_size[-1] = 3

        self.covid_train, self.covid_test = self.load_real_images(covid_path)
        self.healthy_train, self.healthy_test = self.load_real_images(healthy_path)

        self.artificial_covid = self.load_artificial_images(artificial_path + "covid/")
        self.artificial_healthy = self.load_artificial_images(artificial_path + "healthy/")

        self.results_path = results_path

        self.train_images, self.train_labels = self.prepare_train_data()
        self.test_images, self.test_labels = self.prepare_test_data()

        self.artificial_images, self.artificial_labels = self.prepare_artificial_data()

        self.model = self.build_model()

        # Load the pre-trained InceptionV3 model for FID
        self.inception_module = hub.load('https://tfhub.dev/tensorflow/tfgan/eval/inception/1')

        # Initialize the LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex')

    def load_and_vectorize_images(self, image_paths, avg_dimensions):
        images = []
        for image_path in image_paths:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img_resized = img.resize(avg_dimensions)
            img_vector = np.array(img_resized, dtype=np.float32)
            images.append(img_vector)
        return np.stack(images, axis=0)

    def load_artificial_images(self, base_path):
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Directory not found: {base_path}")
        
        all_files = os.listdir(base_path)
        artificial_files = [file for file in all_files if file.endswith('.png')]
        
        if not artificial_files:
            raise ValueError(f"No .png files found in directory: {base_path}")
        
        artificial_paths = [os.path.join(base_path, file) for file in artificial_files]
        
        avg_dimensions = self.image_size[:2]
        artificial_images = self.load_and_vectorize_images(artificial_paths, avg_dimensions)
        
        # Images are already in the correct shape
        artificial_images = artificial_images.astype("float32")
        return artificial_images

    def load_real_images(self, base_path):
        """GAN-specific data preprocessing."""
        all_image_paths = [os.path.join(base_path, image) for image in os.listdir(base_path)]

        # Remove any paths that do not end in .png
        all_image_paths = [path for path in all_image_paths if path.endswith('.png')]

        # Split the image paths into training (first 75%) and testing (last 25%)
        train_index = (len(all_image_paths) * 3) // 4
        train_image_paths = all_image_paths[:train_index]
        test_image_paths = all_image_paths[train_index:]

        avg_dimensions = self.image_size[:2]

        # Load and vectorize images
        train_images = self.load_and_vectorize_images(train_image_paths, avg_dimensions)
        test_images = self.load_and_vectorize_images(test_image_paths, avg_dimensions)
        
        # Images are already in the correct shape
        train_images = train_images.astype("float32")
        test_images = test_images.astype("float32")

        return train_images, test_images

    def prepare_train_data(self):
        """Combines healthy and covid training data and creates labels."""
        # Create labels for training data: Healthy = 0, Covid = 1
        healthy_train_labels = np.zeros(self.healthy_train.shape[0])
        covid_train_labels = np.ones(self.covid_train.shape[0])

        # Combine the healthy and covid training images and labels
        combined_train_images = np.concatenate((self.healthy_train, self.covid_train), axis=0)
        combined_train_labels = np.concatenate((healthy_train_labels, covid_train_labels), axis=0)

        # One-hot encode the labels
        combined_train_labels = to_categorical(combined_train_labels, num_classes=2)

        return combined_train_images, combined_train_labels

    def prepare_test_data(self):
        """Combines healthy and covid test data and creates labels."""
        # Create labels for test data: Healthy = 0, Covid = 1
        healthy_test_labels = np.zeros(self.healthy_test.shape[0])
        covid_test_labels = np.ones(self.covid_test.shape[0])

        # Combine the healthy and covid test images and labels
        combined_test_images = np.concatenate((self.healthy_test, self.covid_test), axis=0)
        combined_test_labels = np.concatenate((healthy_test_labels, covid_test_labels), axis=0)

        # One-hot encode the labels
        combined_test_labels = to_categorical(combined_test_labels, num_classes=2)

        return combined_test_images, combined_test_labels

    def prepare_artificial_data(self):
        """Combines artificial healthy and covid data and creates labels."""
        # Create labels for artificial data: Healthy = 0, Covid = 1
        artificial_healthy_labels = np.zeros(self.artificial_healthy.shape[0])
        artificial_covid_labels = np.ones(self.artificial_covid.shape[0])

        # Combine the artificial healthy and covid images and labels
        combined_artificial_images = np.concatenate((self.artificial_healthy, self.artificial_covid), axis=0)
        combined_artificial_labels = np.concatenate((artificial_healthy_labels, artificial_covid_labels), axis=0)

        # One-hot encode the labels
        combined_artificial_labels = to_categorical(combined_artificial_labels, num_classes=2)

        return combined_artificial_images, combined_artificial_labels

    def build_model(self):
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=tuple(self.image_size)
        )

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model

    def get_inception_features(self, images, batch_size=32):
        # Ensure the images are in the expected range [-1, 1]
        images = (images / 127.5) - 1
        features_list = []
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            outputs = self.inception_module(batch)
            batch_features = outputs['pool_3']
            batch_features = tf.squeeze(batch_features, axis=[1, 2])
            features_list.append(batch_features)
        
        # Concatenate features from all batches
        features = tf.concat(features_list, axis=0)
        return features

    def calculate_fid(self, real_images, generated_images, batch_size=32):
        real_features = self.get_inception_features(real_images, batch_size=batch_size)
        generated_features = self.get_inception_features(generated_images, batch_size=batch_size)
        fid_score = tfgan.eval.frechet_classifier_distance_from_activations(real_features, generated_features)
        return fid_score

    def preprocess_images_for_fid(self, images):
        # Convert images to float32
        images = tf.cast(images, tf.float32)

        # Resize images to 299x299 for InceptionV3
        images_resized = tf.image.resize(images, [299, 299])
        
        # Normalize images to be in the range [-1, 1]
        images_normalized = (images_resized / 127.5) - 1
        return images_normalized

    def calculate_lpips_batch(self, batch_set1, batch_set2):
        assert batch_set1.shape[0] == batch_set2.shape[0], "Batch sizes do not match"
        with torch.no_grad():
            distance = self.lpips_model(batch_set1, batch_set2)
        return distance.mean().item()

    def compute_statistics(self):
        """Computes FID and LPIPS between various image sets."""
        # Prepare real images
        # Combine train and test images for real images
        real_covid_images = np.concatenate((self.covid_train, self.covid_test), axis=0)
        real_healthy_images = np.concatenate((self.healthy_train, self.healthy_test), axis=0)

        # Split real covid images into two halves
        mid_covid = real_covid_images.shape[0] // 2
        real_covid_images_1 = real_covid_images[:mid_covid]
        real_covid_images_2 = real_covid_images[mid_covid:]

        # Split real healthy images into two halves
        mid_healthy = real_healthy_images.shape[0] // 2
        real_healthy_images_1 = real_healthy_images[:mid_healthy]
        real_healthy_images_2 = real_healthy_images[mid_healthy:]

        # Helper function to preprocess images
        def preprocess_for_fid(images):
            images_tf = tf.convert_to_tensor(images)
            images_processed = self.preprocess_images_for_fid(images_tf)
            return images_processed

        # Preprocess images for FID
        real_covid_images_processed = preprocess_for_fid(real_covid_images)
        real_healthy_images_processed = preprocess_for_fid(real_healthy_images)
        real_covid_images_1_processed = preprocess_for_fid(real_covid_images_1)
        real_covid_images_2_processed = preprocess_for_fid(real_covid_images_2)
        real_healthy_images_1_processed = preprocess_for_fid(real_healthy_images_1)
        real_healthy_images_2_processed = preprocess_for_fid(real_healthy_images_2)
        artificial_covid_images_processed = preprocess_for_fid(self.artificial_covid)
        artificial_healthy_images_processed = preprocess_for_fid(self.artificial_healthy)

        # Compute FID scores
        # Baseline FID for covid: between two halves of real covid images
        fid_covid_baseline = self.calculate_fid(real_covid_images_1_processed, real_covid_images_2_processed)

        # Baseline FID for healthy: between two halves of real healthy images
        fid_healthy_baseline = self.calculate_fid(real_healthy_images_1_processed, real_healthy_images_2_processed)

        # FID between real covid and real healthy images
        fid_covid_vs_healthy = self.calculate_fid(real_covid_images_processed, real_healthy_images_processed)

        # FID between real and artificial images
        fid_covid = self.calculate_fid(real_covid_images_processed, artificial_covid_images_processed)
        fid_healthy = self.calculate_fid(real_healthy_images_processed, artificial_healthy_images_processed)

        # Print FID scores
        print(f'Baseline FID score within covid (split halves): {fid_covid_baseline}')
        print(f'Baseline FID score within healthy (split halves): {fid_healthy_baseline}')
        print(f'FID score between real covid and real healthy images: {fid_covid_vs_healthy}')
        print(f'FID score covid vs artificial covid: {fid_covid}')
        print(f'FID score healthy vs artificial healthy: {fid_healthy}')

        # Prepare images for LPIPS
        # Images are already RGB, so we can use them directly
        real_covid_images_rgb = real_covid_images
        real_healthy_images_rgb = real_healthy_images
        real_covid_images_1_rgb = real_covid_images_1
        real_covid_images_2_rgb = real_covid_images_2
        real_healthy_images_1_rgb = real_healthy_images_1
        real_healthy_images_2_rgb = real_healthy_images_2
        artificial_covid_images_rgb = self.artificial_covid
        artificial_healthy_images_rgb = self.artificial_healthy

        # Convert numpy arrays to PyTorch tensors and reshape to [N, C, H, W]
        def to_torch_tensor(images_rgb):
            return torch.tensor(images_rgb).permute(0, 3, 1, 2).float()

        real_covid_images_tensor = to_torch_tensor(real_covid_images_rgb)
        real_healthy_images_tensor = to_torch_tensor(real_healthy_images_rgb)
        real_covid_images_1_tensor = to_torch_tensor(real_covid_images_1_rgb)
        real_covid_images_2_tensor = to_torch_tensor(real_covid_images_2_rgb)
        real_healthy_images_1_tensor = to_torch_tensor(real_healthy_images_1_rgb)
        real_healthy_images_2_tensor = to_torch_tensor(real_healthy_images_2_rgb)
        artificial_covid_images_tensor = to_torch_tensor(artificial_covid_images_rgb)
        artificial_healthy_images_tensor = to_torch_tensor(artificial_healthy_images_rgb)

        # Normalize images to [-1, 1]
        def normalize_images(images_tensor):
            return (images_tensor / 127.5) - 1

        real_covid_images_tensor = normalize_images(real_covid_images_tensor)
        real_healthy_images_tensor = normalize_images(real_healthy_images_tensor)
        real_covid_images_1_tensor = normalize_images(real_covid_images_1_tensor)
        real_covid_images_2_tensor = normalize_images(real_covid_images_2_tensor)
        real_healthy_images_1_tensor = normalize_images(real_healthy_images_1_tensor)
        real_healthy_images_2_tensor = normalize_images(real_healthy_images_2_tensor)
        artificial_covid_images_tensor = normalize_images(artificial_covid_images_tensor)
        artificial_healthy_images_tensor = normalize_images(artificial_healthy_images_tensor)

        # For LPIPS, ensure the number of images is the same
        # Baseline LPIPS within covid
        num_covid_samples = min(real_covid_images_1_tensor.shape[0], real_covid_images_2_tensor.shape[0])
        idx_covid_1 = torch.randperm(real_covid_images_1_tensor.shape[0])[:num_covid_samples]
        idx_covid_2 = torch.randperm(real_covid_images_2_tensor.shape[0])[:num_covid_samples]
        real_covid_images_1_sampled = real_covid_images_1_tensor[idx_covid_1]
        real_covid_images_2_sampled = real_covid_images_2_tensor[idx_covid_2]

        lpips_distance_covid_baseline = self.calculate_lpips_batch(real_covid_images_1_sampled, real_covid_images_2_sampled)

        # Baseline LPIPS within healthy
        num_healthy_samples = min(real_healthy_images_1_tensor.shape[0], real_healthy_images_2_tensor.shape[0])
        idx_healthy_1 = torch.randperm(real_healthy_images_1_tensor.shape[0])[:num_healthy_samples]
        idx_healthy_2 = torch.randperm(real_healthy_images_2_tensor.shape[0])[:num_healthy_samples]
        real_healthy_images_1_sampled = real_healthy_images_1_tensor[idx_healthy_1]
        real_healthy_images_2_sampled = real_healthy_images_2_tensor[idx_healthy_2]

        lpips_distance_healthy_baseline = self.calculate_lpips_batch(real_healthy_images_1_sampled, real_healthy_images_2_sampled)

        # LPIPS between real covid and real healthy images
        num_samples_covid_healthy = min(real_covid_images_tensor.shape[0], real_healthy_images_tensor.shape[0])
        idx_covid = torch.randperm(real_covid_images_tensor.shape[0])[:num_samples_covid_healthy]
        idx_healthy = torch.randperm(real_healthy_images_tensor.shape[0])[:num_samples_covid_healthy]
        real_covid_images_sampled = real_covid_images_tensor[idx_covid]
        real_healthy_images_sampled = real_healthy_images_tensor[idx_healthy]

        lpips_distance_covid_vs_healthy = self.calculate_lpips_batch(real_covid_images_sampled, real_healthy_images_sampled)

        # LPIPS between real and artificial images
        # For covid
        num_covid_samples_artificial = min(artificial_covid_images_tensor.shape[0], real_covid_images_tensor.shape[0])
        idx_covid_real = torch.randperm(real_covid_images_tensor.shape[0])[:num_covid_samples_artificial]
        real_covid_images_sampled = real_covid_images_tensor[idx_covid_real]
        artificial_covid_images_sampled = artificial_covid_images_tensor[:num_covid_samples_artificial]
        lpips_distance_covid = self.calculate_lpips_batch(real_covid_images_sampled, artificial_covid_images_sampled)

        # For healthy
        num_healthy_samples_artificial = min(artificial_healthy_images_tensor.shape[0], real_healthy_images_tensor.shape[0])
        idx_healthy_real = torch.randperm(real_healthy_images_tensor.shape[0])[:num_healthy_samples_artificial]
        real_healthy_images_sampled = real_healthy_images_tensor[idx_healthy_real]
        artificial_healthy_images_sampled = artificial_healthy_images_tensor[:num_healthy_samples_artificial]
        lpips_distance_healthy = self.calculate_lpips_batch(real_healthy_images_sampled, artificial_healthy_images_sampled)

        # Print LPIPS scores
        print(f'Baseline LPIPS Distance within covid (split halves): {lpips_distance_covid_baseline}')
        print(f'Baseline LPIPS Distance within healthy (split halves): {lpips_distance_healthy_baseline}')
        print(f'LPIPS Distance between real covid and real healthy images: {lpips_distance_covid_vs_healthy}')
        print(f'LPIPS Distance covid vs artificial covid: {lpips_distance_covid}')
        print(f'LPIPS Distance healthy vs artificial healthy: {lpips_distance_healthy}')

        # Return the computed statistics
        statistics = {
            'fid_covid_baseline': fid_covid_baseline.numpy(),
            'fid_healthy_baseline': fid_healthy_baseline.numpy(),
            'fid_covid_vs_healthy': fid_covid_vs_healthy.numpy(),
            'fid_covid': fid_covid.numpy(),
            'fid_healthy': fid_healthy.numpy(),
            'lpips_covid_baseline': lpips_distance_covid_baseline,
            'lpips_healthy_baseline': lpips_distance_healthy_baseline,
            'lpips_covid_vs_healthy': lpips_distance_covid_vs_healthy,
            'lpips_covid': lpips_distance_covid,
            'lpips_healthy': lpips_distance_healthy
        }

        return statistics

    def train_and_evaluate(self, use_artificial_data=False):
        """Trains the model and evaluates it on the test set."""
        if use_artificial_data:
            # Combine the real and artificial training images and labels
            combined_train_images = np.concatenate((self.train_images, self.artificial_images), axis=0)
            combined_train_labels = np.concatenate((self.train_labels, self.artificial_labels), axis=0)
        else:
            combined_train_images = self.train_images
            combined_train_labels = self.train_labels

        # Shuffle the training data
        train_indices = np.arange(combined_train_images.shape[0])
        np.random.shuffle(train_indices)
        combined_train_images = combined_train_images[train_indices]
        combined_train_labels = combined_train_labels[train_indices]

        # Normalize the images to [-1, 1] for training
        combined_train_images = (combined_train_images - 127.5) / 127.5
        test_images_normalized = (self.test_images - 127.5) / 127.5

        early_stopping = EarlyStopping(
            monitor='val_loss',    # Metric to monitor
            patience=5,            # Number of epochs with no improvement before stopping
            restore_best_weights=True  # Restore the best weights after stopping
        )

        # Train the model
        self.model.fit(
            combined_train_images, combined_train_labels,
            epochs=100,  # Adjust epochs based on your needs
            batch_size=32,  # Adjust batch size based on your hardware capacity
            validation_data=(test_images_normalized, self.test_labels),
            verbose=2,
            callbacks=[early_stopping]
        )

        # Evaluate the model on the test data
        test_loss, test_accuracy = self.model.evaluate(test_images_normalized, self.test_labels, verbose=2)

        return test_loss, test_accuracy

    def run(self):
        """Runs training first with real data, then with real + artificial data, and writes results to a file."""
        # Compute statistics before training
        statistics = self.compute_statistics()

        # Train with just real data and evaluate
        loss_real, acc_real = self.train_and_evaluate(use_artificial_data=False)

        # Reset the model for a fresh start
        self.model = self.build_model()

        # Train with real + artificial data and evaluate
        loss_combined, acc_combined = self.train_and_evaluate(use_artificial_data=True)

        # Save results to file
        with open(self.results_path, 'w') as f:
            f.write(f"Baseline FID and LPIPS Statistics within Real Images (split halves):\n")
            f.write(f"Baseline FID score within COVID (split halves): {statistics['fid_covid_baseline']}\n")
            f.write(f"Baseline FID score within Healthy (split halves): {statistics['fid_healthy_baseline']}\n")
            f.write(f"Baseline LPIPS Distance within COVID (split halves): {statistics['lpips_covid_baseline']}\n")
            f.write(f"Baseline LPIPS Distance within Healthy (split halves): {statistics['lpips_healthy_baseline']}\n\n")

            f.write(f"FID and LPIPS between Real COVID and Real Healthy Images:\n")
            f.write(f"FID score between Real COVID and Real Healthy: {statistics['fid_covid_vs_healthy']}\n")
            f.write(f"LPIPS Distance between Real COVID and Real Healthy: {statistics['lpips_covid_vs_healthy']}\n\n")

            f.write(f"FID and LPIPS between Artificial and Real Images:\n")
            f.write(f"FID score COVID vs Artificial COVID: {statistics['fid_covid']}\n")
            f.write(f"FID score Healthy vs Artificial Healthy: {statistics['fid_healthy']}\n")
            f.write(f"LPIPS Distance COVID vs Artificial COVID: {statistics['lpips_covid']}\n")
            f.write(f"LPIPS Distance Healthy vs Artificial Healthy: {statistics['lpips_healthy']}\n\n")

            f.write(f"Results with real data only:\n")
            f.write(f"Test Loss: {loss_real}\n")
            f.write(f"Test Accuracy: {acc_real}\n\n")
            f.write(f"Results with real + artificial data:\n")
            f.write(f"Test Loss: {loss_combined}\n")
            f.write(f"Test Accuracy: {acc_combined}\n")

        print(f"Results written to {self.results_path}")


# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# import os
# from PIL import Image

# import tensorflow_gan as tfgan
# import tensorflow_hub as hub
# import lpips
# import torch
# from tensorflow.keras.mixed_precision import set_global_policy

# class ImageEvaluation:
#     def __init__(self, healthy_path, covid_path, artificial_path, results_path, image_size):
#         # set_global_policy('mixed_float16')

#         self.image_size = list(image_size)
#         # self.image_size[-1] = 3
#         self.covid_train, self.covid_test = self.load_real_images(covid_path)
#         self.healthy_train, self.healthy_test = self.load_real_images(healthy_path)

#         self.artificial_covid = self.load_artificial_images(artificial_path + "covid/")
#         self.artificial_healthy = self.load_artificial_images(artificial_path + "healthy/")

#         self.results_path = results_path

#         self.train_images, self.train_labels = self.prepare_train_data()
#         self.test_images, self.test_labels = self.prepare_test_data()

#         self.artificial_images, self.artificial_labels = self.prepare_artificial_data()

#         self.model = self.build_model()

#         # Load the pre-trained InceptionV3 model for FID
#         self.inception_module = hub.load('https://tfhub.dev/tensorflow/tfgan/eval/inception/1')

#         # Initialize the LPIPS model
#         self.lpips_model = lpips.LPIPS(net='alex')

#     def load_and_vectorize_images(self, image_paths, avg_dimensions):
#         images = []
#         for image_path in image_paths:
#             img = Image.open(image_path)
#             img = img.convert('L')
#             img_resized = img.resize(avg_dimensions)
#             img_vector = np.array(img_resized, dtype=np.float32)
#             images.append(img_vector)
#         return np.stack(images, axis=0)

#     def load_artificial_images(self, base_path):
#         if not os.path.exists(base_path):
#             raise FileNotFoundError(f"Directory not found: {base_path}")
        
#         all_files = os.listdir(base_path)
#         artificial_files = [file for file in all_files if file.endswith('.png')]
        
#         if not artificial_files:
#             raise ValueError(f"No .png files found in directory: {base_path}")
        
#         artificial_paths = [os.path.join(base_path, file) for file in artificial_files]
        
#         avg_dimensions = self.image_size[:2]
#         artificial_images = self.load_and_vectorize_images(artificial_paths, avg_dimensions)
        
#         artificial_images = artificial_images.reshape(artificial_images.shape[0], *self.image_size).astype("float32")
#         return artificial_images

#     def load_real_images(self, base_path):
#         """GAN-specific data preprocessing."""
#         # List patient directories
#         patient_dirs = sorted([d for d in os.listdir(base_path) if 'Patient' in d])
#         print("Patient dirs", patient_dirs)

#         train_dirs_index = (len(patient_dirs) * 3) // 4

#         # Select first 60% for training and next 40% for testing
#         train_dirs = patient_dirs[:train_dirs_index]
#         test_dirs = patient_dirs[train_dirs_index:]

#         print("Training directories:", train_dirs)
#         print("Testing directories:", test_dirs)

#         # List all image paths
#         train_image_paths = [os.path.join(base_path, patient, image) 
#                              for patient in train_dirs 
#                              for image in os.listdir(os.path.join(base_path, patient))]

#         test_image_paths = [os.path.join(base_path, patient, image) 
#                             for patient in test_dirs 
#                             for image in os.listdir(os.path.join(base_path, patient))]

#         # Remove any paths that do not end in .png
#         train_image_paths = [path for path in train_image_paths if path.endswith('.png')]
#         test_image_paths = [path for path in test_image_paths if path.endswith('.png')]
#         avg_dimensions = self.image_size[:2]

#         # Load and vectorize images
#         train_images = self.load_and_vectorize_images(train_image_paths, avg_dimensions)
#         test_images = self.load_and_vectorize_images(test_image_paths, avg_dimensions)
        
#         train_images = train_images.reshape(train_images.shape[0], *self.image_size).astype("float32")

#         test_images = test_images.reshape(test_images.shape[0], *self.image_size).astype("float32")

#         return train_images, test_images

#     def prepare_train_data(self):
#         """Combines healthy and covid training data and creates labels."""
#         # Create labels for training data: Healthy = 0, Covid = 1
#         healthy_train_labels = np.zeros(self.healthy_train.shape[0])
#         covid_train_labels = np.ones(self.covid_train.shape[0])

#         # Combine the healthy and covid training images and labels
#         combined_train_images = np.concatenate((self.healthy_train, self.covid_train), axis=0)
#         combined_train_labels = np.concatenate((healthy_train_labels, covid_train_labels), axis=0)

#         # One-hot encode the labels
#         combined_train_labels = to_categorical(combined_train_labels, num_classes=2)

#         return combined_train_images, combined_train_labels

#     def prepare_test_data(self):
#         """Combines healthy and covid test data and creates labels."""
#         # Create labels for test data: Healthy = 0, Covid = 1
#         healthy_test_labels = np.zeros(self.healthy_test.shape[0])
#         covid_test_labels = np.ones(self.covid_test.shape[0])

#         # Combine the healthy and covid test images and labels
#         combined_test_images = np.concatenate((self.healthy_test, self.covid_test), axis=0)
#         combined_test_labels = np.concatenate((healthy_test_labels, covid_test_labels), axis=0)

#         # One-hot encode the labels
#         combined_test_labels = to_categorical(combined_test_labels, num_classes=2)

#         return combined_test_images, combined_test_labels

#     def prepare_artificial_data(self):
#         """Combines artificial healthy and covid data and creates labels."""
#         # Create labels for artificial data: Healthy = 0, Covid = 1
#         artificial_healthy_labels = np.zeros(self.artificial_healthy.shape[0])
#         artificial_covid_labels = np.ones(self.artificial_covid.shape[0])

#         # Combine the artificial healthy and covid images and labels
#         combined_artificial_images = np.concatenate((self.artificial_healthy, self.artificial_covid), axis=0)
#         combined_artificial_labels = np.concatenate((artificial_healthy_labels, artificial_covid_labels), axis=0)

#         # One-hot encode the labels
#         combined_artificial_labels = to_categorical(combined_artificial_labels, num_classes=2)

#         return combined_artificial_images, combined_artificial_labels

#     def build_model(self):
#         """Builds the ResNet50 model."""
#         base_model = ResNet50(
#             include_top=False,  # We will add our own classification layer
#             weights='imagenet',  # Start with random weights, not pretrained on ImageNet
#             input_shape=self.image_size
#         )

#         # Add custom top layers for classification
#         x = base_model.output
#         x = GlobalAveragePooling2D()(x)
#         x = Dense(1024, activation='relu')(x)
#         predictions = Dense(2, activation='softmax')(x)  # 2 classes: Healthy and Covid

#         # Create the full model
#         model = Model(inputs=base_model.input, outputs=predictions)

#         model.compile(optimizer=Adam(learning_rate=0.001),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

#         return model

#     def get_inception_features(self, images, batch_size=32):
#         # Ensure the images are in the expected range [-1, 1]
#         images = (images / 127.5) - 1
#         features_list = []
        
#         # Process images in batches
#         for i in range(0, len(images), batch_size):
#             batch = images[i:i + batch_size]
#             outputs = self.inception_module(batch)
#             batch_features = outputs['pool_3']
#             batch_features = tf.squeeze(batch_features, axis=[1, 2])
#             features_list.append(batch_features)
        
#         # Concatenate features from all batches
#         features = tf.concat(features_list, axis=0)
#         return features

#     def calculate_fid(self, real_images, generated_images, batch_size=32):
#         real_features = self.get_inception_features(real_images, batch_size=batch_size)
#         generated_features = self.get_inception_features(generated_images, batch_size=batch_size)
#         fid_score = tfgan.eval.frechet_classifier_distance_from_activations(real_features, generated_features)
#         return fid_score

#     def preprocess_images_for_fid(self, images):
#         # Convert images to float32
#         images = tf.cast(images, tf.float32)
        
#         # Convert images to RGB by repeating the grayscale channel thrice
#         images_rgb = tf.image.grayscale_to_rgb(images)
        
#         # Resize images to 299x299 for InceptionV3
#         images_resized = tf.image.resize(images_rgb, [299, 299])
        
#         # Normalize images to be in the range [-1, 1]
#         images_normalized = (images_resized / 127.5) - 1
#         return images_normalized

#     def to_rgb(self, images):
#         # Convert images from [N, H, W, 1] to [N, H, W, 3] by repeating the grayscale channel
#         images_rgb = np.repeat(images, 3, axis=-1)
#         return images_rgb

#     def calculate_lpips_batch(self, batch_set1, batch_set2):
#         assert batch_set1.shape[0] == batch_set2.shape[0], "Batch sizes do not match"
#         with torch.no_grad():
#             distance = self.lpips_model(batch_set1, batch_set2)
#         return distance.mean().item()


#     def compute_statistics(self):
#         """Computes FID and LPIPS between various image sets."""
#         # Prepare real images
#         # Combine train and test images for real images
#         real_covid_images = np.concatenate((self.covid_train, self.covid_test), axis=0)
#         real_healthy_images = np.concatenate((self.healthy_train, self.healthy_test), axis=0)

#         # Split real covid images into two halves
#         mid_covid = real_covid_images.shape[0] // 2
#         real_covid_images_1 = real_covid_images[:mid_covid]
#         real_covid_images_2 = real_covid_images[mid_covid:]

#         # Split real healthy images into two halves
#         mid_healthy = real_healthy_images.shape[0] // 2
#         real_healthy_images_1 = real_healthy_images[:mid_healthy]
#         real_healthy_images_2 = real_healthy_images[mid_healthy:]

#         # Helper function to preprocess images
#         def preprocess_for_fid(images):
#             images_tf = tf.convert_to_tensor(images)
#             images_processed = self.preprocess_images_for_fid(images_tf)
#             return images_processed

#         # Preprocess images for FID
#         real_covid_images_processed = preprocess_for_fid(real_covid_images)
#         real_healthy_images_processed = preprocess_for_fid(real_healthy_images)
#         real_covid_images_1_processed = preprocess_for_fid(real_covid_images_1)
#         real_covid_images_2_processed = preprocess_for_fid(real_covid_images_2)
#         real_healthy_images_1_processed = preprocess_for_fid(real_healthy_images_1)
#         real_healthy_images_2_processed = preprocess_for_fid(real_healthy_images_2)
#         artificial_covid_images_processed = preprocess_for_fid(self.artificial_covid)
#         artificial_healthy_images_processed = preprocess_for_fid(self.artificial_healthy)

#         # Compute FID scores
#         # Baseline FID for covid: between two halves of real covid images
#         fid_covid_baseline = self.calculate_fid(real_covid_images_1_processed, real_covid_images_2_processed)

#         # Baseline FID for healthy: between two halves of real healthy images
#         fid_healthy_baseline = self.calculate_fid(real_healthy_images_1_processed, real_healthy_images_2_processed)

#         # FID between real covid and real healthy images
#         fid_covid_vs_healthy = self.calculate_fid(real_covid_images_processed, real_healthy_images_processed)

#         # FID between real and artificial images
#         fid_covid = self.calculate_fid(real_covid_images_processed, artificial_covid_images_processed)
#         fid_healthy = self.calculate_fid(real_healthy_images_processed, artificial_healthy_images_processed)

#         # Print FID scores
#         print(f'Baseline FID score within covid (split halves): {fid_covid_baseline}')
#         print(f'Baseline FID score within healthy (split halves): {fid_healthy_baseline}')
#         print(f'FID score between real covid and real healthy images: {fid_covid_vs_healthy}')
#         print(f'FID score covid vs artificial covid: {fid_covid}')
#         print(f'FID score healthy vs artificial healthy: {fid_healthy}')

#         # Prepare images for LPIPS
#         # Convert images to RGB
#         real_covid_images_rgb = self.to_rgb(real_covid_images)
#         real_healthy_images_rgb = self.to_rgb(real_healthy_images)
#         real_covid_images_1_rgb = self.to_rgb(real_covid_images_1)
#         real_covid_images_2_rgb = self.to_rgb(real_covid_images_2)
#         real_healthy_images_1_rgb = self.to_rgb(real_healthy_images_1)
#         real_healthy_images_2_rgb = self.to_rgb(real_healthy_images_2)
#         artificial_covid_images_rgb = self.to_rgb(self.artificial_covid)
#         artificial_healthy_images_rgb = self.to_rgb(self.artificial_healthy)

#         # Convert numpy arrays to PyTorch tensors and reshape to [N, C, H, W]
#         def to_torch_tensor(images_rgb):
#             return torch.tensor(images_rgb).permute(0, 3, 1, 2).float()

#         real_covid_images_tensor = to_torch_tensor(real_covid_images_rgb)
#         real_healthy_images_tensor = to_torch_tensor(real_healthy_images_rgb)
#         real_covid_images_1_tensor = to_torch_tensor(real_covid_images_1_rgb)
#         real_covid_images_2_tensor = to_torch_tensor(real_covid_images_2_rgb)
#         real_healthy_images_1_tensor = to_torch_tensor(real_healthy_images_1_rgb)
#         real_healthy_images_2_tensor = to_torch_tensor(real_healthy_images_2_rgb)
#         artificial_covid_images_tensor = to_torch_tensor(artificial_covid_images_rgb)
#         artificial_healthy_images_tensor = to_torch_tensor(artificial_healthy_images_rgb)

#         # Normalize images to [-1, 1]
#         def normalize_images(images_tensor):
#             return (images_tensor / 127.5) - 1

#         real_covid_images_tensor = normalize_images(real_covid_images_tensor)
#         real_healthy_images_tensor = normalize_images(real_healthy_images_tensor)
#         real_covid_images_1_tensor = normalize_images(real_covid_images_1_tensor)
#         real_covid_images_2_tensor = normalize_images(real_covid_images_2_tensor)
#         real_healthy_images_1_tensor = normalize_images(real_healthy_images_1_tensor)
#         real_healthy_images_2_tensor = normalize_images(real_healthy_images_2_tensor)
#         artificial_covid_images_tensor = normalize_images(artificial_covid_images_tensor)
#         artificial_healthy_images_tensor = normalize_images(artificial_healthy_images_tensor)

#         # For LPIPS, ensure the number of images is the same
#         # Baseline LPIPS within covid
#         num_covid_samples = min(real_covid_images_1_tensor.shape[0], real_covid_images_2_tensor.shape[0])
#         idx_covid_1 = torch.randperm(real_covid_images_1_tensor.shape[0])[:num_covid_samples]
#         idx_covid_2 = torch.randperm(real_covid_images_2_tensor.shape[0])[:num_covid_samples]
#         real_covid_images_1_sampled = real_covid_images_1_tensor[idx_covid_1]
#         real_covid_images_2_sampled = real_covid_images_2_tensor[idx_covid_2]

#         lpips_distance_covid_baseline = self.calculate_lpips_batch(real_covid_images_1_sampled, real_covid_images_2_sampled)

#         # Baseline LPIPS within healthy
#         num_healthy_samples = min(real_healthy_images_1_tensor.shape[0], real_healthy_images_2_tensor.shape[0])
#         idx_healthy_1 = torch.randperm(real_healthy_images_1_tensor.shape[0])[:num_healthy_samples]
#         idx_healthy_2 = torch.randperm(real_healthy_images_2_tensor.shape[0])[:num_healthy_samples]
#         real_healthy_images_1_sampled = real_healthy_images_1_tensor[idx_healthy_1]
#         real_healthy_images_2_sampled = real_healthy_images_2_tensor[idx_healthy_2]

#         lpips_distance_healthy_baseline = self.calculate_lpips_batch(real_healthy_images_1_sampled, real_healthy_images_2_sampled)

#         # LPIPS between real covid and real healthy images
#         num_samples_covid_healthy = min(real_covid_images_tensor.shape[0], real_healthy_images_tensor.shape[0])
#         idx_covid = torch.randperm(real_covid_images_tensor.shape[0])[:num_samples_covid_healthy]
#         idx_healthy = torch.randperm(real_healthy_images_tensor.shape[0])[:num_samples_covid_healthy]
#         real_covid_images_sampled = real_covid_images_tensor[idx_covid]
#         real_healthy_images_sampled = real_healthy_images_tensor[idx_healthy]

#         lpips_distance_covid_vs_healthy = self.calculate_lpips_batch(real_covid_images_sampled, real_healthy_images_sampled)

#         # Existing LPIPS between real and artificial images
#         # For covid
#         num_covid_samples_artificial = min(artificial_covid_images_tensor.shape[0], real_covid_images_tensor.shape[0])
#         idx_covid_real = torch.randperm(real_covid_images_tensor.shape[0])[:num_covid_samples_artificial]
#         real_covid_images_sampled = real_covid_images_tensor[idx_covid_real]
#         artificial_covid_images_sampled = artificial_covid_images_tensor[:num_covid_samples_artificial]
#         lpips_distance_covid = self.calculate_lpips_batch(real_covid_images_sampled, artificial_covid_images_sampled)

#         # For healthy
#         num_healthy_samples_artificial = min(artificial_healthy_images_tensor.shape[0], real_healthy_images_tensor.shape[0])
#         idx_healthy_real = torch.randperm(real_healthy_images_tensor.shape[0])[:num_healthy_samples_artificial]
#         real_healthy_images_sampled = real_healthy_images_tensor[idx_healthy_real]
#         artificial_healthy_images_sampled = artificial_healthy_images_tensor[:num_healthy_samples_artificial]
#         lpips_distance_healthy = self.calculate_lpips_batch(real_healthy_images_sampled, artificial_healthy_images_sampled)

#         # Print LPIPS scores
#         print(f'Baseline LPIPS Distance within covid (split halves): {lpips_distance_covid_baseline}')
#         print(f'Baseline LPIPS Distance within healthy (split halves): {lpips_distance_healthy_baseline}')
#         print(f'LPIPS Distance between real covid and real healthy images: {lpips_distance_covid_vs_healthy}')
#         print(f'LPIPS Distance covid vs artificial covid: {lpips_distance_covid}')
#         print(f'LPIPS Distance healthy vs artificial healthy: {lpips_distance_healthy}')

#         # Return the computed statistics
#         statistics = {
#             'fid_covid_baseline': fid_covid_baseline.numpy(),
#             'fid_healthy_baseline': fid_healthy_baseline.numpy(),
#             'fid_covid_vs_healthy': fid_covid_vs_healthy.numpy(),
#             'fid_covid': fid_covid.numpy(),
#             'fid_healthy': fid_healthy.numpy(),
#             'lpips_covid_baseline': lpips_distance_covid_baseline,
#             'lpips_healthy_baseline': lpips_distance_healthy_baseline,
#             'lpips_covid_vs_healthy': lpips_distance_covid_vs_healthy,
#             'lpips_covid': lpips_distance_covid,
#             'lpips_healthy': lpips_distance_healthy
#         }

#         return statistics

#     # def compute_statistics(self):
#     #     """Computes FID and LPIPS between various image sets using batching for memory efficiency."""
#     #     # Prepare real images by combining training and testing sets
#     #     real_covid_images = np.concatenate((self.covid_train, self.covid_test), axis=0)
#     #     real_healthy_images = np.concatenate((self.healthy_train, self.healthy_test), axis=0)

#     #     # Split real covid and healthy images into two halves for FID baseline calculations
#     #     mid_covid = real_covid_images.shape[0] // 2
#     #     real_covid_images_1 = real_covid_images[:mid_covid]
#     #     real_covid_images_2 = real_covid_images[mid_covid:]
        
#     #     mid_healthy = real_healthy_images.shape[0] // 2
#     #     real_healthy_images_1 = real_healthy_images[:mid_healthy]
#     #     real_healthy_images_2 = real_healthy_images[mid_healthy:]

#     #     # Helper function to preprocess images for FID calculation
#     #     def preprocess_for_fid(images):
#     #         images_tf = tf.convert_to_tensor(images)
#     #         images_processed = self.preprocess_images_for_fid(images_tf)
#     #         return images_processed

#     #     # Preprocess images for FID in batches
#     #     batch_size = 8  # Adjust based on available GPU memory

#     #     real_covid_images_processed = preprocess_for_fid(real_covid_images)
#     #     real_healthy_images_processed = preprocess_for_fid(real_healthy_images)
#     #     real_covid_images_1_processed = preprocess_for_fid(real_covid_images_1)
#     #     real_covid_images_2_processed = preprocess_for_fid(real_covid_images_2)
#     #     real_healthy_images_1_processed = preprocess_for_fid(real_healthy_images_1)
#     #     real_healthy_images_2_processed = preprocess_for_fid(real_healthy_images_2)
#     #     artificial_covid_images_processed = preprocess_for_fid(self.artificial_covid)
#     #     artificial_healthy_images_processed = preprocess_for_fid(self.artificial_healthy)

#     #     # Compute FID scores with batch processing
#     #     # Baseline FID for covid: between two halves of real covid images
#     #     fid_covid_baseline = self.calculate_fid(real_covid_images_1_processed, real_covid_images_2_processed, batch_size=batch_size)

#     #     # Baseline FID for healthy: between two halves of real healthy images
#     #     fid_healthy_baseline = self.calculate_fid(real_healthy_images_1_processed, real_healthy_images_2_processed, batch_size=batch_size)

#     #     # FID between real covid and real healthy images
#     #     fid_covid_vs_healthy = self.calculate_fid(real_covid_images_processed, real_healthy_images_processed, batch_size=batch_size)

#     #     # FID between real and artificial images
#     #     fid_covid = self.calculate_fid(real_covid_images_processed, artificial_covid_images_processed, batch_size=batch_size)
#     #     fid_healthy = self.calculate_fid(real_healthy_images_processed, artificial_healthy_images_processed, batch_size=batch_size)

#     #     # Print FID scores
#     #     print(f'Baseline FID score within covid (split halves): {fid_covid_baseline}')
#     #     print(f'Baseline FID score within healthy (split halves): {fid_healthy_baseline}')
#     #     print(f'FID score between real covid and real healthy images: {fid_covid_vs_healthy}')
#     #     print(f'FID score covid vs artificial covid: {fid_covid}')
#     #     print(f'FID score healthy vs artificial healthy: {fid_healthy}')

#     #     # LPIPS computations are left as they were, as LPIPS calculations are already batched in the original code.

#     #     # Return the computed statistics
#     #     statistics = {
#     #         'fid_covid_baseline': fid_covid_baseline.numpy(),
#     #         'fid_healthy_baseline': fid_healthy_baseline.numpy(),
#     #         'fid_covid_vs_healthy': fid_covid_vs_healthy.numpy(),
#     #         'fid_covid': fid_covid.numpy(),
#     #         'fid_healthy': fid_healthy.numpy()
#     #     }

#     #     return statistics


#     def train_and_evaluate(self, use_artificial_data=False):
#         """Trains the model and evaluates it on the test set."""
#         if use_artificial_data:
#             # Combine the real and artificial training images and labels
#             combined_train_images = np.concatenate((self.train_images, self.artificial_images), axis=0)
#             combined_train_labels = np.concatenate((self.train_labels, self.artificial_labels), axis=0)
#         else:
#             combined_train_images = self.train_images
#             combined_train_labels = self.train_labels

#         # Shuffle the training data
#         train_indices = np.arange(combined_train_images.shape[0])
#         np.random.shuffle(train_indices)
#         combined_train_images = combined_train_images[train_indices]
#         combined_train_labels = combined_train_labels[train_indices]

#         # Normalize the images to [-1, 1] for training
#         combined_train_images = (combined_train_images - 127.5) / 127.5
#         self.test_images = (self.test_images - 127.5) / 127.5

#         # Train the model
#         self.model.fit(
#             combined_train_images, combined_train_labels,
#             epochs=40,  # Adjust epochs based on your needs
#             batch_size=32,  # Adjust batch size based on your hardware capacity
#             validation_data=(self.test_images, self.test_labels),
#             verbose=2
#         )

#         # Evaluate the model on the test data
#         test_loss, test_accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=2)

#         return test_loss, test_accuracy

#     def run(self):
#         """Runs training first with real data, then with real + artificial data, and writes results to a file."""
#         # Compute statistics before training
#         statistics = self.compute_statistics()

#         # Train with just real data and evaluate
#         loss_real, acc_real = self.train_and_evaluate(use_artificial_data=False)

#         # Reset the model for a fresh start
#         self.model = self.build_model()

#         # Train with real + artificial data and evaluate
#         loss_combined, acc_combined = self.train_and_evaluate(use_artificial_data=True)

#         # Save results to file
#         with open(self.results_path, 'w') as f:
#             f.write(f"Baseline FID and LPIPS Statistics within Real Images (split halves):\n")
#             f.write(f"Baseline FID score within COVID (split halves): {statistics['fid_covid_baseline']}\n")
#             f.write(f"Baseline FID score within Healthy (split halves): {statistics['fid_healthy_baseline']}\n")
#             f.write(f"Baseline LPIPS Distance within COVID (split halves): {statistics['lpips_covid_baseline']}\n")
#             f.write(f"Baseline LPIPS Distance within Healthy (split halves): {statistics['lpips_healthy_baseline']}\n\n")

#             f.write(f"FID and LPIPS between Real COVID and Real Healthy Images:\n")
#             f.write(f"FID score between Real COVID and Real Healthy: {statistics['fid_covid_vs_healthy']}\n")
#             f.write(f"LPIPS Distance between Real COVID and Real Healthy: {statistics['lpips_covid_vs_healthy']}\n\n")

#             f.write(f"FID and LPIPS between Artificial and Real Images:\n")
#             f.write(f"FID score COVID vs Artificial COVID: {statistics['fid_covid']}\n")
#             f.write(f"FID score Healthy vs Artificial Healthy: {statistics['fid_healthy']}\n")
#             f.write(f"LPIPS Distance COVID vs Artificial COVID: {statistics['lpips_covid']}\n")
#             f.write(f"LPIPS Distance Healthy vs Artificial Healthy: {statistics['lpips_healthy']}\n\n")

#             f.write(f"Results with real data only:\n")
#             f.write(f"Test Loss: {loss_real}\n")
#             f.write(f"Test Accuracy: {acc_real}\n\n")
#             f.write(f"Results with real + artificial data:\n")
#             f.write(f"Test Loss: {loss_combined}\n")
#             f.write(f"Test Accuracy: {acc_combined}\n")

#         print(f"Results written to {self.results_path}")
