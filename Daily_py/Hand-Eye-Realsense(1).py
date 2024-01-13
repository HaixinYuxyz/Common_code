import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os

K = np.array([[388.343, 0, 324.99],
              [0, 388.343, 238.259],
              [0, 0, 1]], dtype=np.float64)  # 相机内参
chess_board_x_num = 12  # 棋盘格x方向格子数
chess_board_y_num = 9  # 棋盘格y方向格子数
chess_board_len = 30  # 单位棋盘格长度,mm


# 用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


# 用于根据位姿计算变换矩阵
def pose_robot(list):
    Tx = list[0] * 1000
    Ty = list[1] * 1000
    Tz = list[2] * 1000
    x = list[3]
    y = list[4]
    z = list[5]
    thetaX = x  # / 180 * pi
    thetaY = y  # / 180 * pi
    thetaZ = z  # / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    # RT1=np.linalg.inv(RT1)
    return RT1


# 用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path, chess_board_x_num, chess_board_y_num, K, chess_board_len):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    img = cv2.imread(img_path)
    # print(img)
    # print(os.path.exists(img_path))
    # cv2.imshow('color_img', img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
    # print(corners)
    # if corners is not None:
    #     print("good")
    corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
    for i in range(corners.shape[0]):
        corner_points[:, i] = corners[i, 0, :]
    # print(corner_points)
    object_points = np.zeros((3, chess_board_x_num * chess_board_y_num), dtype=np.float64)
    flag = 0
    for i in range(chess_board_y_num):
        for j in range(chess_board_x_num):
            object_points[:2, flag] = np.array([(11 - j - 1) * chess_board_len, (8 - i - 1) * chess_board_len])
            flag += 1
    # print(object_points)
    #
    # cv2.imshow('color_img', img)
    # cv2.waitKey(0)

    retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, K, distCoeffs=None)
    # print(rvec.reshape((1,3)))
    # RT=np.column_stack((rvec,tvec))
    RT = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    # RT=pose(rvec[0,0],rvec[1,0],rvec[2,0],tvec[0,0],tvec[1,0],tvec[2,0])
    # print(RT)

    # print(retval, rvec, tvec)
    # print(RT)
    # print('')
    return RT


folder = r"C:\Users\lenovo\Desktop\realsense2UR_calibData"  # 棋盘格图片存放文件夹
# files = os.listdir(folder)
# file_num=len(files)
# RT_all=np.zeros((4,4,file_num))

# print(get_RT_from_chessboard('calib/2.bmp', chess_board_x_num, chess_board_y_num, K, chess_board_len))
'''
这个地方很奇怪的特点，有些棋盘格点检测得出来，有些检测不了，可以通过函数get_RT_from_chessboard的运行时间来判断
'''
# good_picture = [1, 3, 4, 7, 8]  # 存放可以检测出棋盘格角点的图片
good_picture = [1, 3, 4, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18]
# good_picture = [2, 5, 6, 8, 14]
file_num = len(good_picture)

pose_data = []
pose_data.append([-0.019078476029065677, -0.369125193568132, 0.7701693852815668, -0.691131750395322, 1.5048891660179067, -1.528870691858659])
pose_data.append([0.03260771096928429, -0.34393400880207825, 0.791765096936394, -0.9502465099648617, 0.8818134952874166, -1.5111193020718716])
pose_data.append([-0.030871028858295665, -0.27658430686938273, 0.7467941186521937, -0.9069809576375957, 1.6384009040763363, -1.9478906604444948])
pose_data.append([0.021326271439168792, -0.39007226351048174, 0.8552266330416823, -0.9338859031562143, 0.8520525140377666, -1.472633756546116])
pose_data.append([0.01209349341088823, -0.4512139235744426, 0.8199739438785433, -0.8637193003742174, 0.8943299532099908, -1.4122946798273268])

pose_data.append([-0.05694857425433267, -0.5188841973257059, 0.7796374738668925, -0.5255815922363785, -0.2775690263031889, -2.2121268431605356])
pose_data.append([-0.27730062581848214, -0.4964612690917667, 0.8384103524662173, 0.12595131778411, 0.1601034427917842, -1.7105970344653938])
pose_data.append([-0.06407049894296732, -0.3751115763742816, 0.8475275258906649, -0.13162245613607051, 1.4328357725155378, -1.363720945272301])
pose_data.append([0.2364969581611217, -0.688502288852702, 0.6193773075665372, -0.42010335829807843, 0.3640621743639397, -2.256312270889589])
pose_data.append([0.27709324660670004, -0.6137654130001803, 0.6892888039036396, -0.557539312433207, 0.11579591901488598, -2.679794467391061])
pose_data.append([0.6802971820052015, -0.5165049553345423, 0.384623776913535, -0.07130166902458983, 0.5857455072266485, -1.9076969628848992])
pose_data.append([0.47852563364752765, -0.6747774800319769, 0.45900125446108453, -0.8868159865668193, 0.0926277275416002, -2.1739206557977067])
pose_data.append([0.2395363153548839, -0.5951475831546577, 0.7174511265835528, 0.7916322433117251, 0.025397842868970472, -2.37815932983409])

# pose_data.append()

# pose_data.append([0.23950168074867936, -0.5951648383703338, 0.7174398451949386, 0.7916626923369753, 0.025432917640018683, -2.3783203592080757])
# pose_data.append([0.02161431159923103, -0.6414250896365581, 0.7361831255138873, -0.12065857357703695, -0.28755227746744805, -1.8153344303565027])
# pose_data.append([-0.11374706586692454, -0.597564010999371, 0.8118353551188837, -0.024829168514751067, -0.29223696544475475, -2.6851340581166614])
# pose_data.append([-0.11235198151228837, -0.5997372763375239, 0.7076561802951742, 0.15621574807921873, -0.02553742802344048, -2.1680772551633654])
# pose_data.append([-0.01683177217325963, -0.34194176599421305, 0.7721722073448642, 0.6822656034968522, 0.45380612664223113, -2.1040052474524975])
# pose_data.append([-0.04544338174555086, -0.40188696404986707, 0.7707385442437661, 0.3953273135273054, 0.000692553205713495, -1.2855351762146772])
# pose_data.append([-0.17582015902093404, -0.41452627612050597, 0.8219988372840055, 0.6041097539492654, -0.1359897825511189, -1.1428105069167955])
# pose_data.append([0.15432100846925947, -0.464243315637907, 0.7360534292494066, 0.8553380113501571, 0.822772026451759, -1.3702198817277107])
# pose_data.append([0.2038212755856318, -0.36585955848839086, 0.9000478436091488, -0.12065031207574266, -0.33630237824322456, -1.2960344033218343])
# pose_data.append([0.28367996726124767, -0.7122967215993338, 0.5668193217003494, -0.16745580057159845, 0.7529651579324071, -1.2215033933871022])
# pose_data.append([0.2214460498480971, -0.6210840190627548, 0.6853690303046837, -0.6588811982348487, -0.5602717972130112, -1.4140901171449545])
# pose_data.append([0.73002701930022, -0.03415324281345657, 0.6359843147350958, -0.48658669709458746, 0.3021857936037081, -1.8046386626934563])
# pose_data.append([0.6612130348288637, -0.19731340296211408, 0.6681886076352385, 0.2872859937493143, -0.7168778507061165, -2.0962595075211694])
# pose_data.append([0.3898647607736422, -0.36550020196954836, 0.8538532956439828, -0.12083891217753583, -0.6842893888928575, -1.4245243170547601])
# pose_data.append([0.79115428840163, -0.39623337078803794, 0.2781945897307808, 0.1995271330963691, -0.20916738095718432, -2.3572500898928586])

# 计算board to cam 变换矩阵
R_all_chess_to_cam_1 = []
T_all_chess_to_cam_1 = []
# RT = []
# RT4 = []
for i in good_picture:
    print(i)
    image_path = folder + '\\' + str(i) + '_Color.png'
    RT = get_RT_from_chessboard(image_path, chess_board_x_num - 1, chess_board_y_num - 1, K, chess_board_len)

    R_all_chess_to_cam_1.append(RT[:3, :3])
    T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3, 1)))
    RT4 = RT
# print(T_all_chess_to_cam.shape)

# 计算end to base变换矩阵
# file_address = 'E:\\Research\\AAA_BallCatching\\realsense2UR_calibData\\机器人基坐标位姿.xlsx'  # 从记录文件读取机器人六个位姿
# sheet_1 = pd.read_excel(file_address)
R_all_base_to_end_1 = []
T_all_base_to_end_1 = []
# print(sheet_1.iloc[0]['ax'])
ind = 0
for i in good_picture:
    # print(sheet_1.iloc[i-1]['ax'],sheet_1.iloc[i-1]['ay'],sheet_1.iloc[i-1]['az'],sheet_1.iloc[i-1]['dx'],
    #                                   sheet_1.iloc[i-1]['dy'],sheet_1.iloc[i-1]['dz'])
    # RT = np.linalg.inv(pose_robot(pose_data[ind]))
    RT = pose_robot(pose_data[ind])
    # RT=np.column_stack(((cv2.Rodrigues(np.array([[sheet_1.iloc[i-1]['ax']],[sheet_1.iloc[i-1]['ay']],[sheet_1.iloc[i-1]['az']]])))[0],
    #                    np.array([[sheet_1.iloc[i-1]['dx']],
    #                                   [sheet_1.iloc[i-1]['dy']],[sheet_1.iloc[i-1]['dz']]])))
    # RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    # RT = np.linalg.inv(RT)

    R_all_base_to_end_1.append(RT[:3, :3])
    T_all_base_to_end_1.append(RT[:3, 3].reshape((3, 1)))
    ind = ind + 1

# print(R_all_end_to_base_1)
R, T = cv2.calibrateHandEye(R_all_base_to_end_1, T_all_base_to_end_1, R_all_chess_to_cam_1,
                            T_all_chess_to_cam_1)  # 手眼标定
RT = np.column_stack((R, T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))  # 即为cam to base变换矩阵
print('相机相对于基座的变换矩阵为：')
print(np.linalg.inv(RT))

# 结果验证，原则上来说，每次结果相差较小
for i in range(len(good_picture)):
    print('第', i, '次')

    RT_base_to_End = np.column_stack((R_all_base_to_end_1[i], T_all_base_to_end_1[i]))
    RT_base_to_End = np.row_stack((RT_base_to_End, np.array([0, 0, 0, 1])))
    # print(RT_base_to_End)

    RT_chess_to_cam = np.column_stack((R_all_chess_to_cam_1[i], T_all_chess_to_cam_1[i]))
    RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
    # print(RT_chess_to_cam)

    RT_cam_to_base = np.column_stack((R, T))
    RT_cam_to_base = np.row_stack((RT_cam_to_base, np.array([0, 0, 0, 1])))
    # print(RT_cam_to_base)

    RT_chess_to_end = RT_base_to_End @ RT_cam_to_base @ RT_chess_to_cam  # 即为固定的棋盘格相对于机器人末端坐标系位姿
    # RT_chess_to_end = np.linalg.inv(RT_chess_to_end)

    print(RT_chess_to_end[:3, :])
    print('')
