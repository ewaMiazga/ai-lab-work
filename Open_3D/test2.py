import open3d as o3d
import numpy as np

# Load depth image and normal map
depth = o3d.io.read_image("depth.jpg")
normals = o3d.io.read_image("normals.jpg")

# Convert depth to point cloud
pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, o3d.camera.PinholeCameraIntrinsic())

# Assign normal data
pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))

# Poisson Reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Visualize
o3d.visualization.draw_geometries([mesh])
