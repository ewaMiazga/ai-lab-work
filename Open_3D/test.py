import torch
import cv2
import open3d as o3d
import numpy as np
import torchvision.transforms as T
from torchvision import models

# Download the image, "cube_resized_16big.png"
depth = cv2.imread("depth_map_im0.png", cv2.IMREAD_UNCHANGED)
# Depth is 4 dim but I need to use only 3 of them but contract them to one dim

# Convert the image to grayscale if it has multiple channels
if depth.ndim == 3:
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

if depth.dtype != np.uint16:
    depth = (depth.astype(np.float32) / 255.0 * 65535).astype(np.uint16)

    # Save the image as 16-bit PNG
depth_img_path = "16bit.png"
cv2.imwrite(depth_img_path, depth)

# Load the depth image using Open3D
depth_map = o3d.io.read_image(depth_img_path)

print(depth.dtype)

#depth = depth[:, :, 1]
# Define camera intrinsic parameters
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=672, height=384, fx=1758.23, fy=1758.23, cx=953.34, cy=552.29
)

# Create point cloud from depth image
pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsic)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
# print(depth.shape)

# # Convert depth to point cloud and visualize in Open3D
# height, width = depth.shape
# fx, fy = 1758.23, 1758.23
# cx, cy = 953.34 / 2, 552.29 / 2

# points = []
# for v in range(height):
#     for u in range(width):
#         z = depth[v, u]
#         if z > 0:
#             x = (u - cx) * z / fx
#             y = (v - cy) * z / fy
#             points.append([x, y, z])

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(points))
# o3d.visualization.draw_geometries([pcd])
