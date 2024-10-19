import numpy as np
import gc

def load_dataset(path):
    gc.collect()
    images_path = f"{path}/Images(500x500).npy"
    writer_info_path = f"{path}/WriterInfo.npy"

    images = np.load(images_path)
    writer_info = np.load(writer_info_path)
    print(f"Loaded {images.shape[0]} images and {writer_info.shape[0]} writer information entries.")
    return images, writer_info

def create_targets(writer_info):
    return writer_info[:, 0]