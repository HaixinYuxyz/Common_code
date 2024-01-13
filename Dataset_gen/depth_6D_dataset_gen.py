import glob
import os
import json
import cv2
import numpy as np
from plyfile import PlyData
import pandas as pd
import shutil
import random
from tqdm import tqdm
from stl import mesh


def read_stl(path):
    your_mesh = mesh.Mesh.from_file(path)
    point = your_mesh.v0
    return point


def verification_obj(target_path, obj_name, mode, if_random):
    # point_cloud = read_ply(r'G:\transparent_6D_pose_estimate\Dataset\one_obj_dataset_900+3600\mmmmm1\model_simple.ply')
    point_cloud = read_stl(r'G:\Dataset_general_materials\变换坐标系+归一\zhengfamin_pingdi.stl')

    sample_num = int(0.002 * point_cloud.shape[0])  # 假设取50%的数据
    sample_list = [i for i in range(point_cloud.shape[0])]  # [0, 1, 2, 3]
    sample_list = random.sample(sample_list, sample_num)  # [1, 2]

    point_cloud = point_cloud[sample_list, :]

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


def verification_hand_keypoint(target_path, mode, if_random):
    pic_num = len(glob.glob(os.path.join(target_path, mode, 'rgb', '*')))
    for k in range(807, pic_num):
        if if_random:
            k = random.randint(1, pic_num)
        print('--------> showing pic {}'.format(k))
        json_data = json.load(open(os.path.join(target_path, mode, 'log_json', '{}.json'.format(k))))
        img = cv2.imread(os.path.join(target_path, mode, 'rgb', '{}.png'.format(k)))
        camera_point = np.array(json_data['camera point'])
        camera_angle = np.array(json_data['camera angle'])
        camera_K = np.array(json_data['camera internal']).reshape((3, 3))
        hand_keypoint = np.array(list(json_data['hand_keypoint'].values())).reshape(21, 3)
        if hand_keypoint.shape[1] == 3:
            hand_keypoint = np.hstack((hand_keypoint, np.ones((hand_keypoint.shape[0], 1))))
        T_matrix_list = np.array([])
        for i in range(21):
            T_matrix = cal_T(camera_point, camera_angle, hand_keypoint[i, :-1], np.array([0, 0, 0]))
            T_matrix_list = np.append(T_matrix_list, T_matrix)
        T_matrix_list = T_matrix_list.reshape((21, 4, 4))
        for p in range(21):
            point_cam = T_matrix_list[p].dot(np.array([0, 0, 0, 1]).T).T.reshape(1, 4)
            point_cam = point_cam[:, :3]
            uv = camera_K.dot(point_cam.T)
            pix = uv.T / uv.T[:, 2:3]
            pix = pix[:, :2]
            cv2.circle(img, (int(pix[0][0]), int(pix[0][1])), radius=2, color=(0, 0, 255))
        cv2.imshow("Image", img)
        cv2.waitKey(0)


def point_proj(xyz, cam_K):
    uv = cam_K.dot(xyz.T)
    pix = uv.T / uv.T[:, 2:3]
    pix = pix[:, :2]
    return pix


def point_2_rgb(img, pix):
    num = pix.shape[0]
    for i in range(num):
        cv2.circle(img, (int(pix[i][0]), int(pix[i][1])), radius=2, color=(187, 104, 76))
    return img


# 通过角度计算旋转矩阵
def AnglesTorotationMatrix(angle):
    x = angle[0] * np.pi / 180
    y = angle[1] * np.pi / 180
    z = angle[2] * np.pi / 180
    R = np.array([[np.cos(z) * np.cos(y), np.cos(z) * np.sin(y) * np.sin(x) - np.sin(z) * np.cos(x),
                   np.cos(z) * np.sin(y) * np.cos(x) + np.sin(z) * np.sin(x)],
                  [np.sin(z) * np.cos(y), np.sin(z) * np.sin(y) * np.sin(x) + np.cos(z) * np.cos(x),
                   np.sin(z) * np.sin(y) * np.cos(x) - np.cos(z) * np.sin(x)],
                  [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x)]])
    return R


# 计算整体的旋转矩阵
def cal_T(camera_point, camera_angle, obj_position, obj_rotation):
    cal = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R_cam_world = AnglesTorotationMatrix(camera_angle)
    t_cam_world = camera_point
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R_cam_world
    T_cam_world[:3, 3] = t_cam_world
    print(T_cam_world)
    R_obj_world = AnglesTorotationMatrix(obj_rotation)
    t_obj_world = obj_position
    T_obj_world = np.eye(4)
    T_obj_world[:3, :3] = R_obj_world
    T_obj_world[:3, 3] = t_obj_world
    print(T_obj_world)

    T_matrix = np.linalg.inv(T_cam_world).dot(T_obj_world)
    T_matrix = cal.dot(T_matrix)

    return T_matrix


def move_rename_file(file_path_root, target_path, hand_name, obj_name, pic_per_pose):
    """
    将文件转移并重命名
    :param file_path_root:
    :param target_path:
    :param hand_name:
    :param obj_name:
    :param pic_per_pose:
    """
    file_list = glob.glob(os.path.join(file_path_root, '*'))
    for mode in ['pose', 'depth']:
        num_log = 0
        for file in file_list:
            for i in tqdm(range(pic_per_pose - 1)):
                shutil.copy(os.path.join(file, mode, 'image---{}.png0000.png'.format(i)),
                            os.path.join(target_path, mode, 'rgb', '{}.png'.format(num_log)))
                if mode == 'depth':
                    shutil.copy(os.path.join(file, mode, '{}.npy'.format(i + 1)),
                                os.path.join(target_path, mode, 'depth_np', '{}.npy'.format(num_log)))
                else:
                    shutil.copy(os.path.join(file, mode, '{}.npy'.format(i)),
                                os.path.join(target_path, mode, 'depth_np', '{}.npy'.format(num_log)))
                shutil.copy(os.path.join(file, mode, '{}.json'.format(i)),
                            os.path.join(target_path, mode, 'log_json', '{}.json'.format(num_log)))
                shutil.copy(os.path.join(file, mode, 'mask{}---{}0000.png'.format(obj_name, i)),
                            os.path.join(target_path, mode, 'obj_mask', '{}.png'.format(num_log)))
                shutil.copy(os.path.join(file, mode, 'mask{}---{}0000.png'.format(hand_name, i)),
                            os.path.join(target_path, mode, 'hand_mask', '{}.png'.format(num_log)))
                shutil.copy(os.path.join(file, mode, 'depth---{}.png0000.png'.format(i)),
                            os.path.join(target_path, mode, 'depth_png', '{}.png'.format(num_log)))
                num_log = num_log + 1
    # os.remove(os.path.join(target_path, 'pose', 'depth_np', '0.npy'))
    # for change_num in range(1, num_log):
    #     os.rename(os.path.join(target_path, 'pose', 'depth_np', '{}.npy'.format(change_num)),
    #               os.path.join(target_path, 'pose', 'depth_np', '{}.npy'.format(change_num - 1)))

    # for mode in ['pose', 'depth']:
    #     os.remove(os.path.join(target_path, mode, 'depth_png', '{}.png'))


def post_process_pose():
    pass


def clean_up_dataset(file_path_root, target_path, hand_name, obj_name, pic_per_pose):
    """
    将文件进行归类
    :param file_path_root: 总文件夹路径
    :param target_path: 目标存放文件夹路径
    :param hand_name: 手对应的名称
    :param obj_name: 物体对应的名称
    :param pic_per_pose: 每个姿态采集的图像数量
    """

    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for mode in ['pose', 'depth']:
        os.mkdir(os.path.join(target_path, mode))
        os.mkdir(os.path.join(target_path, mode, 'obj_mask'))
        os.mkdir(os.path.join(target_path, mode, 'hand_mask'))
        os.mkdir(os.path.join(target_path, mode, 'rgb'))
        os.mkdir(os.path.join(target_path, mode, 'depth_png'))
        os.mkdir(os.path.join(target_path, mode, 'depth_np'))
        os.mkdir(os.path.join(target_path, mode, 'log_json'))

    move_rename_file(file_path_root, target_path, hand_name, obj_name, pic_per_pose)


def make_hand_kpts(target_path):
    for mode in ['pose', 'depth']:
        os.mkdir(os.path.join(target_path, mode, 'hand_kpt'))
        pic_num = len(glob.glob(os.path.join(target_path, mode, 'rgb', '*')))
        for i in tqdm(range(pic_num)):
            json_data = json.load(open(os.path.join(target_path, mode, 'log_json', '{}.json'.format(i))))
            img = cv2.imread(os.path.join(target_path, mode, 'rgb', '{}.png'.format(i)))
            camera_point = np.array(json_data['camera point'])
            camera_angle = np.array(json_data['camera angle'])
            camera_K = np.array(json_data['camera internal']).reshape((3, 3))
            hand_keypoint = np.array(list(json_data['hand_keypoint'].values())).reshape(21, 3)
            if hand_keypoint.shape[1] == 3:
                hand_keypoint = np.hstack((hand_keypoint, np.ones((hand_keypoint.shape[0], 1))))
            T_matrix_list = np.array([])
            for j in range(21):
                T_matrix = cal_T(camera_point, camera_angle, hand_keypoint[j, :-1], np.array([0, 0, 0]))
                T_matrix_list = np.append(T_matrix_list, T_matrix)
            T_matrix_list = T_matrix_list.reshape((21, 4, 4))
            pixs = np.array([])
            for p in range(21):
                point_cam = T_matrix_list[p].dot(np.array([0, 0, 0, 1]).T).T.reshape(1, 4)
                point_cam = point_cam[:, :3]
                uv = camera_K.dot(point_cam.T)
                pix = uv.T / uv.T[:, 2:3]
                pix = pix[:, :2]
                pixs = np.append(pixs, pix)
                cv2.circle(img, (int(pix[0][0]), int(pix[0][1])), radius=2, color=(21, 23, 24))
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            np.savetxt(os.path.join(target_path, mode, 'hand_kpt', '{}.txt'.format(i)), pixs.reshape(-1, 2))


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
    file_path_root = r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\raw_dataset\handobj_10'
    target_path = r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0007'
    # point_cloud = r'G:\Dataset_general_materials\变换坐标系+归一\zhengfamin_yuandi.stl'
    hand_name = r'Man_casual.002'
    obj_name = r'handobj_10'
    pic_per_pose = 1800
    mode = 'pose'
    if_random = False

    # verification_final(target_path, mode, if_random)
    # verification_obj(target_path, obj_name, mode, if_random)
    verification_hand_keypoint(target_path, mode, if_random)

    # clean_up_dataset(file_path_root, target_path, hand_name, obj_name, pic_per_pose)
    # make_hand_kpts(target_path)
