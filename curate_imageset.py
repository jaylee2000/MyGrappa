import os
import random
import shutil

imgs = []

root_directory = './128x128'
# Iterate over subdirectories
for sub_dir in os.listdir(root_directory):
    if (not sub_dir.endswith(".csv")) and ('DS_Store' not in sub_dir):
        sub_dir_path = os.path.join(root_directory, sub_dir)

        # Create train, val, and test subdirectories if they don't exist
        for sub_sub_dir in ['train', 'val', 'test']:
            sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
            if not os.path.exists(sub_sub_dir_path):
                os.makedirs(sub_sub_dir_path)

        # List all image files in the current subdirectory
        image_files = [f for f in os.listdir(sub_dir_path) if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"))]

        # Calculate the number of images for each split
        total_images = len(image_files)
        train_size = int(0.8 * total_images)
        val_size = int(0.1 * total_images)

        # Shuffle the list of image files
        random.shuffle(image_files)

        # Move images to the corresponding subdirectories
        for i, filename in enumerate(image_files):
            src_path = os.path.join(sub_dir_path, filename)
            if i < train_size:
                dst_path = os.path.join(sub_dir_path, 'train', filename)
            elif i < train_size + val_size:
                dst_path = os.path.join(sub_dir_path, 'val', filename)
            else:
                dst_path = os.path.join(sub_dir_path, 'test', filename)

            # Move the file
            shutil.move(src_path, dst_path)

print("Images organized into train, val, and test sets.")