import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib


def gauss_transformation(pos_img, width_img, center):
    center_x, center_y = center[0], center[1]
    IMAGE_HEIGHT = pos_img.shape[0]
    IMAGE_WIDTH = pos_img.shape[1]
    R = 0.5 * 50
    mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
    mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

    x1 = np.arange(IMAGE_WIDTH)
    x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

    y1 = np.arange(IMAGE_HEIGHT)
    y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
    y_map = np.transpose(y_map)

    Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)

    Gauss_map = np.exp(-0.5 * Gauss_map / R)
    guess_img = pos_img * Gauss_map
    return guess_img


def get_mask_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        M["m00"] = 1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel


if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\lenovo\Desktop\65\65\rgb_mask\05555.jpg', cv2.IMREAD_GRAYSCALE)
    thresh, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = find_max_region(img)
    cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = get_mask_center(cnt[0])
    cv2.circle(img, center, 7, (0, 0, 255), -1)
    plt.figure()
    plt.imshow(img, cmap="hot")
    plt.show()


    # img = np.ones((300, 300))
    # img = gauss_transformation(img, 0, (150, 150))
    # plt.figure()
    # plt.imshow(img, cmap="hot")
    # plt.show()
