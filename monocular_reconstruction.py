# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:58:41 2025

@author: shubh
"""
#%% 1.libraries import
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

#%% 2.Getting model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/gLpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/gLpn-nyu")

#%% 3.Loading and resizing the image

image = Image.open("D:/shubham/DATA/weird_bird_AI.jpg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width - diff if diff <16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

#%% 4.Preparing the image for the model

inputs = feature_extractor(images=image, return_tensors="pt")

#%% 5.Getting the prediciton from the model

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    
#%% 6.Post-Processing
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# visualize the prediction
fig, ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output,cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)

#%% 7. Preparing the depth image for open3d

width, height = image.size
depth_image = (output * 255 / np.max(output)).astype('uint8')
image = np.array(image)

# create rgbd image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d,depth_o3d, convert_rgb_to_intensity=False)

#%% 8. Creating a Camera

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

#%% 9. Creating o3d point cloud

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

o3d.visualization.draw_geometries([pcd])

#%% 10. Post-processing the 3D Point cloud

# outliers removal
c1, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
pcd = pcd.select_by_index(ind)

# estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()
o3d.visualization.draw_geometries([pcd])

#%% 11. Surface reconstruction

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

# rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# visualization the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#%% 12. 3D Mesh Export

o3d.io.write_triangle_mesh(r'D:\shubham\RESULTS\bird_AI.ply', mesh)

