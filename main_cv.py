import cv2
import numpy as np

def stereoBM_gpu(left_image_path, right_image_path, num_disparities=64, block_size=21):
    # Load stereo images in grayscale
    img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_left is None or img_right is None:
        raise ValueError("Images not found. Check the file paths.")
    
    # Ensure the OpenCV CUDA module is available
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("CUDA-enabled GPU not detected!")

    # Upload images to the GPU
    gpu_img_left = cv2.cuda_GpuMat()
    gpu_img_right = cv2.cuda_GpuMat()
    gpu_img_left.upload(img_left)
    gpu_img_right.upload(img_right)

    # Create GPU StereoBM object
    stereo = cv2.cuda.createStereoBM(numDisparities=num_disparities, blockSize=block_size)

    # Compute disparity on GPU
    disparity_gpu = stereo.compute(gpu_img_left, gpu_img_right)

    # Download disparity map from GPU to CPU
    disparity = disparity_gpu.download()

    # Normalize disparity for visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # Display the disparity map
    cv2.imshow("Disparity Map", disparity_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disparity

# Example usage
dataset_dir = "dataset/"
# Time measurement
import time
start = time.time()
disparity_map = stereoBM_gpu(f"{dataset_dir}chess1_r.jpg", f"{dataset_dir}chess2_r.jpg")
end = time.time()

print(f"Disparity map computed in {end - start:.2f} seconds.")
