from skimage import color, io
import os
import numpy as np

root_directory = './128x128'
for sub_dir in os.listdir(root_directory):
    if (not sub_dir.endswith(".csv")) and ('DS_Store' not in sub_dir):
        sub_dir_path = os.path.join(root_directory, sub_dir)
        for sub_sub_dir in ['train', 'val', 'test']:
            sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
            image_files = [f for f in os.listdir(sub_sub_dir_path) if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"))]
            for i, filename in enumerate(image_files):
                src_path = os.path.join(sub_sub_dir_path, filename)
                dst_path = os.path.join(sub_sub_dir_path, filename.split('.')[0] + '.npy')
                
                img = io.imread(src_path)
                img_gray = color.rgb2gray(img)
                
                if img_gray.shape != (128, 128):
                    img_gray = np.resize(img_gray, (128, 128))
                np.save(dst_path, img_gray)


