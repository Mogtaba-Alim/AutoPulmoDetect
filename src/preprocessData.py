# Make sure Pillow is installed or run pip install Pillow
import os
from PIL import Image
import subprocess

# Function to check directory and download dataset if needed
def check_and_download_dataset(path_to_raw_data, dataset_identifier):
    if not os.path.exists(path_to_raw_data):
        # Create the RawData directory if it doesn't exist
        os.makedirs(os.path.dirname(path_to_raw_data), exist_ok=True)
        # Use Kaggle API to download the dataset
        # Ensure that your kaggle.json file with your api key is in ~/.kaggle directory
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_identifier, '--path', os.path.dirname(path_to_raw_data), '--unzip'], check=True)

# Define the base directory paths
path_to_raw_data = "../rawData"
base_src_path = os.path.join(path_to_raw_data, "chest_xray")
base_dst_path = os.path.join(path_to_raw_data, "../procData")
dataset_identifier = "paultimothymooney/chest-xray-pneumonia"

# Check if the dataset exists, if not, download it
check_and_download_dataset(base_src_path, dataset_identifier)

# Check if the procData directory exists, if not, create it
if not os.path.exists(base_dst_path):
    print("Creating procData directory")
    os.makedirs(base_dst_path)

# Create the target directory structure if it doesn't exist
for category in ["Normal", "Pneumonia"]:
    os.makedirs(os.path.join(base_dst_path, category), exist_ok=True)

# Function to resize image
def resize_image(src_file, dst_file, size=(224, 224)):
    with Image.open(src_file) as img:
        # Use Image.Resampling.LANCZOS for high-quality downsampling
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        img_resized.save(dst_file)

# Define the categories and subcategories
subdirs = ["test", "train", "val"]
categories = ["NORMAL", "PNEUMONIA"]

# Iterate through each subdirectory and category to copy and resize files
for subdir in subdirs:
    for category in categories:
        src_dir = os.path.join(base_src_path, subdir, category)
        dst_dir = os.path.join(base_dst_path, "Normal" if category == "NORMAL" else "Pneumonia")

        # Copy and resize each file from the source to the destination
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            resize_image(src_file, dst_file)

print("Images have been copied and resized to the procData directory.")
