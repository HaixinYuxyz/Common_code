import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def dilate_image(image, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated


# Function to apply Sobel operator
def apply_sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    return magnitude


# Function to apply Canny edge detector
def apply_canny(image):
    edges = cv2.Canny(image, 3, 10)
    return edges


def image_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def make_outline(image):
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的背景图像
    outline = np.zeros_like(image)

    # 画出轮廓
    cv2.drawContours(outline, contours, -1, (255, 255, 255), 2)
    return outline


if __name__ == '__main__':
    # rgb_image = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0005\pose\rgb\453.png')
    # gray_image = image_to_gray(rgb_image)
    # hand_mask = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0005\pose\hand_mask\453.png', cv2.IMREAD_GRAYSCALE)
    # hand_outline = make_outline(hand_mask)
    # obj_mask = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0005\pose\obj_mask\453.png', cv2.IMREAD_GRAYSCALE)
    # obj_outline = make_outline(obj_mask)
    # gray_image[hand_mask == 0] = 0
    # # gray_image[obj_outline == 255] = 0
    # mask_all = np.logical_or(hand_mask, obj_mask)
    # mask_not = 255 - mask_all
    # image = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0005\pose\rgb\604.png')
    depth = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0007\depth\depth_png\1325.png', cv2.IMREAD_GRAYSCALE)
    mask_obj = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0007\depth\obj_mask\1325.png', cv2.IMREAD_GRAYSCALE)
    no_mask = (255 - mask_obj) / 255

    depth_no_obj = no_mask * depth

    depth_no_obj[depth_no_obj == 0] = 255

    # mask_hand = cv2.imread(r'G:\handhold_transparent_object_depth_completion\Dataset\HandPoseDataset\processed_dataset\0005\pose\hand_mask\604.png', cv2.IMREAD_GRAYSCALE)
    # mask_all = np.logical_or(mask_obj, mask_hand)
    # image = image_to_gray(image)
    # depth = apply_canny(depth)

    # depth = dilate_image(depth, kernel_size=5, iterations=2)
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_no_obj)
    plt.show()
