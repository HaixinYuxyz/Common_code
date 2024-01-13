import cv2
from stl import mesh
import numpy as np
from plyfile import PlyData
import pandas as pd
import glob
import random
import json
import os
from depth_6D_dataset_gen import cal_T




def read_ply(filename):
    plydata = PlyData.read(filename)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]
    point = data_np[:, :3]
    point = point.astype(np.float32)

    return point


def read_stl(path):
    your_mesh = mesh.Mesh.from_file(path)
    point = your_mesh.v0
    return point


def verification_obj(target_path, obj_name, mode, if_random):
    point_cloud = read_ply(r'G:\transparent_6D_pose_estimate\Dataset\one_obj_dataset_900+3600\mmmmm1\model_simple.ply')
    pic_num = len(glob.glob(os.path.join(target_path, mode, 'rgb', '*')))
    for i in range(pic_num):
        if if_random:
            i = random.randint(1, pic_num)
        print('--------> showing pic {}'.format(i))
        json_data = json.load(open(os.path.join(target_path, mode, 'log_json', '{}.json'.format(i))))
        obj_location = np.array(json_data[obj_name]['location'])
        obj_rotation = np.array(json_data[obj_name]['rotation_euler'])
        img = cv2.imread(os.path.join(target_path, mode, 'rgb', '{}.png'.format(i)))
        camera_point = np.array(json_data['camera point'])
        camera_angle = np.array(json_data['camera angle'])
        hand_keypoint = np.array(list(json_data['hand_keypoint'].values())).reshape(21, 3)
        camera_K = np.array(json_data['camera internal']).reshape((3, 3))

        if point_cloud.shape[1] == 3:
            point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        T_matrix = cal_T(camera_point, camera_angle, obj_location, obj_rotation)
        point_cam = T_matrix.dot(point_cloud.T).T
        point_cam = point_cam[:, :3]
        depth = point_cam[:, 2]
        uv = camera_K.dot(point_cam.T)
        pix = uv.T / uv.T[:, 2:3]
        pix = pix[:, :2]
        for p in pix:
            cv2.circle(img, (int(p[0]), int(p[1])), radius=0, color=(255, 0, 0))
        cv2.imshow("Image", img)
        cv2.waitKey(0)


def verification_final(target_path, mode, if_random):
    path = os.path.join(target_path, mode)
    pic_num = len(glob.glob(os.path.join(path, 'rgb', '*')))
    for i in range(1, pic_num):
        if if_random:
            i = random.randint(1, pic_num)
        print('--------> showing pic {}'.format(i))
        rgb = cv2.imread(os.path.join(path, 'rgb', '{}.png'.format(i))) / 255
        h, w, c = rgb.shape
        depth = np.expand_dims(np.load(os.path.join(path, 'depth_np', '{}.npy'.format(i))), axis=2)
        depth = np.concatenate((depth, depth, depth), axis=2)
        hand_mask = cv2.imread(os.path.join(path, 'hand_mask', '{}.png'.format(i)))
        obj_mask = cv2.imread(os.path.join(path, 'obj_mask', '{}.png'.format(i)))
        show_pic = np.zeros((2 * h, 2 * w, c))
        show_pic[0:h, 0:w, :] += rgb
        show_pic[h:2 * h, 0:w, :] += depth
        show_pic[0:h, w:2 * w, :] += hand_mask
        show_pic[h:2 * h, w:2 * w, :] += obj_mask
        show_pic = cv2.resize(show_pic, (w, h))
        cv2.imshow('win', show_pic)
        cv2.waitKey(0)


if __name__ == '__main__':
    obj_path = r'G:\Dataset_general_materials\变换坐标系+归一\shaobei.stl'
    stl = read_stl(obj_path)
    # print(stl)

    
    file_path_root = r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\raw_dataset\handobj_6'
    target_path = r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0005'
    hand_name = r'Man_casual.002'
    obj_name = r'handobj_6'
    pic_per_pose = 181
    mode = 'pose  '
    if_random = True

    verification_final(target_path, mode, if_random)