"""
Create image dataset for training, validation, and testing.
Original downloaded dataset from https://www.kaggle.com/datasets/rhtsingh/google-universal-image-embeddings-128x128/data
"""

import os
import random
from shutil import move, rmtree

def main():
    module = 'test' # 'val' or 'test'
    NUMBER = 100 # 100

    # Set the paths to your image and numpy file folders
    root_dir = os.path.join('/storage/jeongjae/128x128/landmark', module)
    image_folder = os.path.join(root_dir, 'img_rgb')
    npy_folder = os.path.join(root_dir, 'npy_bw')

    # Ensure both folders exist
    if not os.path.exists(image_folder) or not os.path.exists(npy_folder):
        print("Image or numpy folders do not exist.")
        exit()

    # Get the list of file numbers
    file_numbers = [file.split('image')[1].split('.jpg')[0] for file in os.listdir(image_folder)]
    file_numbers.sort()

    # Select 1000 random numbers to keep
    numbers_to_keep = random.sample(file_numbers, NUMBER)
    numbers_to_keep.sort()

    # Move the selected files to a temporary folder
    temp_folder = os.path.join(root_dir, 'temp_keep')
    os.makedirs(temp_folder, exist_ok=True)

    for number in numbers_to_keep:
        image_file = os.path.join(image_folder, f'image{number}.jpg')
        npy_file = os.path.join(npy_folder, f'image{number}.npy')
        
        move(image_file, os.path.join(temp_folder, os.path.basename(image_file)))
        move(npy_file, os.path.join(temp_folder, os.path.basename(npy_file)))

    # Delete the remaining files
    for file in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file)
        os.remove(file_path)

    for file in os.listdir(npy_folder):
        file_path = os.path.join(npy_folder, file)
        os.remove(file_path)

    # Move the selected files back to the original folders
    for file in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file)
        if file.endswith('.jpg'):
            move(file_path, os.path.join(image_folder, file))
        elif file.endswith('.npy'):
            move(file_path, os.path.join(npy_folder, file))

    # Remove the temporary folder
    rmtree(temp_folder)

    print("Files successfully selected and others deleted.")

if __name__ == "__main__":
    main()