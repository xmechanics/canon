import os
import sys
import random
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from canon.autoencode.report import backend_report
backend_report()

from canon.autoencode.tf import reset_tf_session
from canon.autoencode import builder
from canon.autoencode.train_v2 import train

def sample_images(source_folder, dest_folder, sample_size=100):
    """
    Randomly sample up to sample_size images from source_folder and copy them to dest_folder.
    
    Args:
        source_folder (str): Path to the source folder containing images
        dest_folder (str): Path to the destination folder where sampled images will be saved
        sample_size (int): Maximum number of images to sample (default: 100)
    
    Returns:
        int: Number of images actually sampled
    """
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get list of image files in source folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(source_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(source_folder).glob(f'*{ext.upper()}')))
    
    # Determine how many images to sample
    num_to_sample = min(sample_size, len(image_files))
    
    if num_to_sample == 0:
        print(f"No images found in {source_folder}")
        return 0
    
    # Randomly sample the images
    sampled_images = random.sample(image_files, num_to_sample)
    
    # Copy sampled images to destination folder
    for img_path in sampled_images:
        dest_path = os.path.join(dest_folder, img_path.name)
        shutil.copy2(img_path, dest_path)
    
    print(f"Successfully sampled {num_to_sample} images from {source_folder} to {dest_folder}")
    return num_to_sample


def split_images(source_folder, train_folder, val_folder, val_size=1000, total_sample=None):
    """
    Randomly split images from source_folder into training and validation sets.
    
    Args:
        source_folder (str): Path to the source folder containing images
        train_folder (str): Path to the destination folder for training images
        val_folder (str): Path to the destination folder for validation images
        val_size (int): Number of images for validation set (default: 100)
        total_sample (int, optional): Total number of images to sample. If None, use all available images.
    
    Returns:
        tuple: (number of training images, number of validation images)
    """
    # Create destination folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Get list of image files in source folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(source_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(source_folder).glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {source_folder}")
        return 0, 0
        
    # If total_sample is specified, randomly select that many images
    if total_sample is not None:
        total_sample = min(total_sample, len(image_files))
        image_files = random.sample(image_files, total_sample)
    
    # Adjust validation size if it exceeds available images
    val_size = min(val_size, len(image_files))
    
    # Randomly select validation images
    val_images = random.sample(image_files, val_size)
    
    # The rest go to training
    train_images = [img for img in image_files if img not in val_images]
    
    # Copy validation images
    for img_path in val_images:
        dest_path = os.path.join(val_folder, img_path.name)
        shutil.copy2(img_path, dest_path)
    
    # Copy training images
    for img_path in train_images:
        dest_path = os.path.join(train_folder, img_path.name)
        shutil.copy2(img_path, dest_path)
    
    print(f"Split complete: {len(train_images)} training images, {len(val_images)} validation images")
    return len(train_images), len(val_images)


if __name__ == "__main__":
    from canon.common.init import init_logging
    init_logging()
    
    source_folder = os.path.join("..", "data", "mnt", "Laue_8k")
    training_folder = os.path.join("..", "data", "mnt", "training")
    validation_folder = os.path.join("..", "data", "mnt", "validation")
    # split_images(source_folder, training_folder, validation_folder, val_size=1000)

    nersc = ("IN_NERSC" in os.environ) and os.environ["IN_NERSC"] == "true"
    s = reset_tf_session(nersc=nersc)

    backbone = "convnexttiny"
    n_features = 256

    train(backbone, n_features,
          training_folder,
          validation_folder,
          verbose=1,
          epochs=1000,
          dryrun=False, use_generator=True)
