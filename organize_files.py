import os
import random
import shutil

def split_files(input_folder, output_folder, train_percentage=0.8, val_percentage=0.1, test_percentage=0.1):
    # Get a list of all files in the input folder
    all_files = os.listdir(input_folder)

    # Calculate the number of files for each split
    num_files = len(all_files)
    num_train = int(num_files * train_percentage)
    num_val = int(num_files * val_percentage)
    num_test = int(num_files * test_percentage)

    # Shuffle the list of files
    random.shuffle(all_files)

    # Create subfolders if they don't exist
    for folder in ['train', 'val', 'test']:
        subfolder_path = os.path.join(output_folder, folder)
        os.makedirs(subfolder_path, exist_ok=True)

    # Move files to the appropriate subfolders
    for i, file in enumerate(all_files):
        if i < num_train:
            shutil.move(os.path.join(input_folder, file), os.path.join(output_folder, 'train', file))
        elif i < num_train + num_val:
            shutil.move(os.path.join(input_folder, file), os.path.join(output_folder, 'val', file))
        else:
            shutil.move(os.path.join(input_folder, file), os.path.join(output_folder, 'test', file))

if __name__ == '__main__':
    # Example usage:
    input_folder = '/storage/jeongjae/128x128/landmark'
    output_folder = input_folder
    split_files(input_folder, output_folder)
