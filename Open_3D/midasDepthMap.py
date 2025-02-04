import cv2
import torch
import numpy as np
import time

# Load a MiDaS model for depth estimation
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Load the image
img_path = "../dataset/cube_resized.jpg"
img = cv2.imread(img_path)

# Convert the image to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply input transforms
input_batch = transform(img).to(device)

# Prediction and resize to original resolution
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# Normalize the depth map for visualization
depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

# Convert the depth map to 8-bit for visualization
depth_map_8bit = (depth_map * 255).astype(np.uint8)
depth_map_colored = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_MAGMA)

# Save the depth map
cv2.imwrite("cube_resized_depth_map.png", depth_map_8bit)
cv2.imwrite("cube_resized_depth_map_colored.png", depth_map_colored)

# Display the image and depth map
cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imshow('Depth Map', depth_map_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()