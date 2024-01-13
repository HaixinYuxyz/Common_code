import pandas as pd
import numpy as np
import open3d as o3d
import cv2
import OpenEXR
import Imath
from PIL import Image

camera_factor = 1
camera_cx = 1280 / 2
camera_cy = 720 / 2

camera_fx = 1137.77783203125
camera_fy = 1137.77783203125


def read_raw_data(data_path):
    if data_path[-3:] == 'png' or data_path[-3:] == 'jpg':
        data = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    elif data_path[-3:] == 'npy':
        data = np.load(data_path) * 255
    elif data_path[-3:] == 'exr':
        file = OpenEXR.InputFile(data_path)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image = [Image.frombytes("F", size, file.channel(c, pt)) for c in "G"]
        data = np.array(image[0])
    return data


def show_point_cloud(data, if_show):
    h, w = data.shape
    points = np.zeros((w * h, 3), dtype=np.float32)
    n = 0
    for i in range(h):
        for j in range(w):
            deep = data[i, j]
            points[n][0] = j
            points[n][1] = i
            points[n][2] = deep
            n = n + 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if if_show:
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    return pcd


def show_point_cloud_with_ins(data, if_show):
    h, w = data.shape
    points = np.zeros((w * h, 3), dtype=np.float32)
    n = 0
    for i in range(h):
        for j in range(w):
            deep = data[i, j]
            points[n][2] = deep / camera_factor
            points[n][0] = (j - camera_cx) * points[n][2] / camera_fx
            points[n][1] = (i - camera_cy) * points[n][2] / camera_fy
            n = n + 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if if_show:
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    return pcd


def cluster(pcd, if_show):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # 画个框玩玩
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    if if_show:
        o3d.visualization.draw_geometries([outlier_cloud, aabb],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    return pcd


def surface_reconstruction(pcd, type=None):
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    pcd.estimate_normals()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])


if __name__ == '__main__':
    data_path = r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0000\depth\depth_np\190.npy'
    data = read_raw_data(data_path)
    pcd = show_point_cloud_with_ins(data, if_show=False)
    pcd = cluster(pcd, if_show=True)
    # surface_reconstruction(pcd)
