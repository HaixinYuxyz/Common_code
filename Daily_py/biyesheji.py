import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

path = r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG'


def save_fig(data, name):
    plt.close()
    plt.figure()
    plt.imshow(data, cmap='jet')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', dpi=600, pad_inches=0.0)


# def save_fig_with_colorbar(data, name):
#     plt.close()
#     fig, ax = plt.subplots()
#     cax = ax.imshow(data, cmap='jet')
#     plt.axis('off')
#     fig.colorbar(cax, ax=ax, shrink=1, aspect=20, pad=0.01)
#     plt.savefig(name, bbox_inches='tight', dpi=600, pad_inches=0.0)


for i in tqdm(range(21)):
    for j in range(20):
        # 读取Numpy数组数据
        # pred_depth = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_pred_depth.npy'.format(i, j))
        # d_conf = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_d_conf.npy'.format(i, j))
        # d_depth = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_d_depth.npy'.format(i, j))
        # rgb_conf = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_rgb_conf.npy'.format(i, j))
        rgb_depth = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_rgb_depth.npy'.format(i, j))
        # conf_inital = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_conf_inital.npy'.format(i, j))
        # depth_without_objs = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_depth_without_objs.npy'.format(i, j))
        depth_label = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\{}_{}_depths_label.npy'.format(i, j))
        error = np.abs(depth_label - rgb_depth)

        # save_fig(pred_depth, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_pred_depth.png'.format(i, j))
        # save_fig(d_conf, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_d_conf.png'.format(i, j))
        # save_fig(d_depth, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_d_depth.png'.format(i, j))
        # save_fig(rgb_conf, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_rgb_conf.png'.format(i, j))
        # save_fig(rgb_depth, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_rgb_depth.png'.format(i, j))
        # save_fig(conf_inital, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_conf_inital.png'.format(i, j))
        # save_fig(depth_without_objs, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR2\{}_{}_depth_without_objs.png'.format(i, j))
        # save_fig_with_colorbar(depth_label, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\PNG_COLOR\{}_{}_depths_label.png'.format(i, j))
        save_fig(error, r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\Error\{}_{}_error_rgb_depth.png'.format(i, j))
