import os
import kaggle

# Set the Kaggle dataset path
dataset_path = "msambare/fer2013"
download_path = "./datasets"

# Download dataset
kaggle.api.dataset_download_files(dataset_path, path=download_path, unzip=True)

print(f"Dataset downloaded to: {download_path}")
