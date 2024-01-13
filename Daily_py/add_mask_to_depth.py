import cv2
import numpy as np

if __name__ == '__main__':
    depth = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0007\depth\depth_png\80.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0007\depth\obj_mask\80.png', cv2.IMREAD_GRAYSCALE)
    mask = (255 - mask) / 255
    depth = mask * depth
    print(depth)
