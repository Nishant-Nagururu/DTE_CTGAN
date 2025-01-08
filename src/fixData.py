import os
import shutil

# Define source and destination paths
source_path = "/blue/dream_team/CT_GAN_TEAM/DTE_CTGAN/Inputs/New_Data_CoV2/Healthy"
destination_path = "/blue/dream_team/CT_GAN_TEAM/DTE_CTGAN/Inputs/Data/Healthy"

# Ensure the destination path exists
os.makedirs(destination_path, exist_ok=True)

# Initialize a counter for renaming files
file_counter = 1

# Iterate through folders in the source path
for folder_name in os.listdir(source_path):
    folder_path = os.path.join(source_path, folder_name)
    print("folder path", folder_path)
    # Check if the folder name starts with "Patient" and is a directory
    if folder_name.startswith("Patient") and os.path.isdir(folder_path):
        # Iterate through files in the folder
        for file_name in os.listdir(folder_path):
            # Check if the file is a .png image
            if file_name.endswith(".png"):
                file_path = os.path.join(folder_path, file_name)
                
                # Generate a new name for the file
                new_file_name = f"{file_counter}.png"
                new_file_path = os.path.join(destination_path, new_file_name)
                
                # Copy and rename the file
                shutil.copy(file_path, new_file_path)
                print(f"Copied: {file_path} -> {new_file_path}")
                
                # Increment the file counter
                file_counter += 1

print("All images have been copied and renamed.")
