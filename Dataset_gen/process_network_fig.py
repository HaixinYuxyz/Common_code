'''
用来绘制深度补全的网络图的数据处理
'''
import numpy as np
import cv2

if __name__ == '__main__':
    pic = np.load(r'G:\毕业设计\中期\绘图图片\handdepthnet_pred\1_9_pred_depth.npy')
    print(pic)
