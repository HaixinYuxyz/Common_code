import numpy as np
import cv2
import os
from plyfile import PlyData
import pandas as pd


def calc_emb_bp_fast(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
            np.einsum(
                "ijkl,ijlm->ijkm",
                R.T.reshape(1, 1, 3, 3),
                depth.reshape(height, width, 1, 1)
                * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
                - T.reshape(1, 1, 3, 1),
            ).squeeze()
            * mask.reshape(height, width, 1)
    )

    return ProjEmb


def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图


def mask2bbox_xyxy(mask):
    """NOTE: the bottom right point is included"""
    ys, xs = np.nonzero(mask)[:2]
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    return [bb_tl[0], bb_tl[1], bb_br[0], bb_br[1]]


def calc_emb_bp_fast(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)  # 对K求逆矩阵

    height, width = depth.shape  # 求高度和宽度
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
            np.einsum(
                "ijkl,ijlm->ijkm",
                R.T.reshape(1, 1, 3, 3),
                depth.reshape(height, width, 1, 1)
                * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
                - T.reshape(1, 1, 3, 1),
            ).squeeze()
            * mask.reshape(height, width, 1)
    )

    return ProjEmb


if __name__ == '__main__':
    filepath = r'G:\transparent_6D_pose_estimate\Dataset\one_obj_dataset_900+3600'
    obj_name_list = ['budui1']
    for obj in obj_name_list:
        plydata = PlyData.read(os.path.join(filepath, obj, r'model2.ply'))  # 读取文件
        data = plydata.elements[0].data  # 读取数据
        data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
        data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
        property_names = data[0].dtype.names  # 读取property的名字
        for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
            data_np[:, i] = data_pd[name]
        point = np.hstack((data_np, np.ones((data_np.shape[0], 1))))
        for mode in ['train', 'test']:
            if mode == 'train':
                num_data = 900
                # os.mkdir(os.path.join(filepath, obj, mode, r'edge'))
            elif mode == 'test':
                num_data = 3600
                # os.mkdir(os.path.join(filepath, obj, mode, r'edge'))
            for num in range(313, 315):
                print('processing----->{},mode = {},num = {}'.format(obj, mode, num))
                mask = cv2.imread(os.path.join(filepath, obj, mode, r'mask', r'{}.png'.format(num)),
                                  cv2.IMREAD_GRAYSCALE)
                # bboxs = mask_find_bboxs(mask)
                # bbox_visib = [bboxs[0][0], bboxs[0][1], bboxs[0][0] + bboxs[0][2], bboxs[0][1] + bboxs[0][3]]
                x1, y1, x2, y2 = mask2bbox_xyxy(mask)  # 生成的边界框
                K = np.array([568.8889, 0, 320, 0, 568.8889, 240, 0, 0, 1]).reshape((3, 3))
                R_path = os.path.join(filepath, obj, mode, r'pose', r'pose{}.npy'.format(num))
                pose = np.load(R_path).reshape(4, 4)
                R = pose[:3, :3]
                t = pose[:3, 3]
                depthmap = np.zeros((480, 640)) + 255
                point_cam = pose.dot(point.T).T
                point_cam = point_cam[:, :3]
                depth = point_cam[:, 2]
                uv = K.dot(point_cam.T)
                pix = uv.T / uv.T[:, 2:3]
                pix = pix[:, :2]
                with_depth = np.concatenate((pix, depth.reshape(-1, 1)), axis=1)
                for pix___ in with_depth:
                    if depthmap[int(pix___[1]), int(pix___[0])] > pix___[2]:
                        cv2.circle(depthmap, (int(pix___[0]), int(pix___[1])), radius=0, color=pix___[2])
                depthmap[depthmap == 255] = 0
                print(depthmap.max())
                depthmap = depthmap / depthmap.max() * 255
                depthmap = cv2.GaussianBlur(depthmap, (7, 7), 0)
                depthmap = np.array(depthmap, dtype=np.uint8)
                edges = cv2.Canny(depthmap, 10, 50)
                edges[edges != 0] = 255
                # edges = np.repeat(edges,3,0).reshape(480,640,3)
                # edges = np.array(edges, dtype=np.float64)
                edges = cv2.GaussianBlur(edges, (7, 7), 0)
                # edges = cv2.medianBlur(edges, (7,7))
                edges[edges != 0] = 255
                edges = cv2.GaussianBlur(edges, (7, 7), 0)

                # np.save(os.path.join(filepath, obj, mode, r'edge', r'{}.npy'.format(num)), edges)
