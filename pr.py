# 只需修改路径 line250,line252

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps
import scipy
import math
from numpy import trapz
import glob


# 获取每个算法目录
def get_algorithm_dir(bbPath):
    algorithm_path = []
    for root, dirs, files in os.walk(bbPath):
        for dire in dirs:
            algorithm_path.append(os.path.join(root, dire))  # 获取每个跟踪算法的路径
    return algorithm_path


# 获取一个算法中的数据文件
def get_datafile(bbPath):
    for root, dirs, files in os.walk(bbPath):
        pass
    return files


# 计算归一化距离
def getDistance(bbPath, gtPath):
    # dell=['7.txt','114.txt','115.txt','12.txt','129.txt','131.txt','142.txt','145.txt','176.txt','197.txt','320.txt','335.txt','394.txt']
    dell = []
    algorithm = get_algorithm_dir(bbPath)
    distance = {}  # 存放所有跟踪算法的norm distance,字典嵌套列表{[[],[],...] , ... ,[[],[],...]}
    for dire in algorithm:
        algo_name = dire.split('/')[-1]  # 获取算法名
        data_file = get_datafile(dire)  # 获取该算法下面的数据文件path

        algo_distance = []  # 存放单个跟踪算法的norm distance 列表嵌套列表
        for fname in data_file:  # 处理一个算法的数据

            if fname in dell:
                continue
            else:
                bb_file_path = os.path.join(dire, fname)  # 单个数据文件路径,例如1.txt
                gt_file_path = os.path.join(gtPath, fname)

                try:
                    bb_data = np.loadtxt(bb_file_path, dtype=np.float64)
                except ValueError:
                    bb_data = np.loadtxt(bb_file_path, dtype=np.float64, delimiter=',')

                try:
                    gt_data = np.loadtxt(gt_file_path, dtype=np.float64)
                except ValueError:
                    gt_data = np.loadtxt(gt_file_path, dtype=np.float64, delimiter=',')

            seq_distance = []  # seq_distance中存放的是一个文件中所有bbox之间的norm distance

            # 为什么不用zip 有时候可能跟踪的序列数少于groundtruth_rect
            for i in range(len(bb_data)):  # 处理一个序列的数据
                gt_x, gt_y, gt_w, gt_h = gt_data[i]
                bb_x, bb_y, bb_w, bb_h = bb_data[i]

                # gt中心点位置 and bbox中心点位置
                gt_center = np.array([gt_x + gt_w / 2, gt_y + gt_h / 2])  # groundtruth bbox center point position
                bb_center = np.array([bb_x + bb_w / 2, bb_y + bb_h / 2])  # trace algorithm bbox center point position

                dx = gt_center[0] - bb_center[0]
                dy = gt_center[1] - bb_center[1]
                ndistance = math.sqrt(dx ** 2 + dy ** 2)  # compute the distance

                seq_distance.append(ndistance)  # 存放了一个数据文件中所有bbox的distance

            algo_distance.append(seq_distance)  # algo_distance中存放的是单个算法中的所有norm distance
        # print(algo_name, algo_distance)
        distance[algo_name] = algo_distance  # distance是一个字典,key对应算法名,value对应normalized distance
    return distance


# 计算精确度
def calculate_accuracy(threshold, bbPath, gtPath):
    # print("bbPath",bbPath)
    #每个 bb 和 gt 的中心距离
    norm_distance = getDistance(bbPath, gtPath)
    algo_accuracy = {}
    key_list = []
    # print(norm_distance)
    for algo_name, algo_distance in norm_distance.items():
        # print(algo_name)
        accuracy_list = []
        for thre in threshold:
            accuracy = 0
            for ndistance in algo_distance:  # len(algo_distance)相当于一个跟踪算法中的序列数
                cnt = 0
                for dist in ndistance:  # 计算单个算法的accuracy
                    if dist < thre:
                        cnt = cnt + 1
                accuracy = accuracy + cnt / len(ndistance)  # 计算每个序列的平均accuracy len(ndistance)相当于帧数
            accuracy_list.append(accuracy / len(algo_distance))  # 计算算法的平均accuracy
        # print(algo_name, accuracy_list)

        y = np.array(accuracy_list)
        x = np.array(threshold)
        # area = trapz(y, x, dx=0.001) / 50
        area = y[20]
        area = '%.03f' % area  # 保留三位小数
        # print(algo_name[37:]+":"+area)
        algo_accuracy['[' + area + ']' + algo_name] = accuracy_list
        key_list.append('[' + area + ']' + algo_name)
        # print(key_list)
    return algo_accuracy, key_list


# 绘制图片
def plot_figure(threshold, list_accuracy, key_list):  # ,list_name,k
    # 设置图像的大小
    plt.figure(figsize=(10, 10))
    # 设置坐标轴上坐标刻度
    plt.xticks(list(np.arange(0, 51, 5)))
    plt.yticks(list(np.arange(0, 1.0, 0.1)), ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    plt.tick_params(labelsize=13)  # 设置坐标轴刻度字体的大小
    plt.grid(alpha=1, ls='--')  # 设置网格线背景，并设置透明度为1
    plt.axis([0, 50, 0, 0.9])  # 设置坐标轴起始点
    colors = ['red', 'dodgerblue', 'yellow', 'black', 'darkred', 'darkorchid', 'lime', 'cyan', 'slategrey', 'maroon',
              'rosybrown', \
              'coral', 'deeppink', 'tan', 'magenta', 'green', 'pink', 'olive', 'gold', 'plum', 'peru', 'chocolate',
              'crimson', \
              'crimson', 'deepskyblue', 'tan', 'springgreen', 'slategrey', 'plum', 'steelblue', 'lawngreen',
              'royalblue']
    # linestyle
    # linestyles = ['-','--']
    i = 0
    x = np.array(threshold)
    t = len(key_list)
    if t % 2 == 1:
        t = (t + 1) / 2
    else:
        t = t / 2
    for key in key_list:
        print("key", key[0:7] + key[44:])
        acu = list_accuracy[key]
        y = np.array(acu)
        # 对图像进行拟合成光滑的曲线
        x_smooth = np.linspace(x.max(), x.min(), 200)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        if i < t:
            plt.plot(x_smooth, y_smooth, color=colors[i], label=key[0:7] + key[44:], linewidth=6, linestyle='-')
        else:
            plt.plot(x_smooth, y_smooth, color=colors[i], label=key[0:7] + key[44:], linewidth=6, linestyle='--')
        # plt.scatter(x, y)
        i = i + 1

    # 设置图例的属性
    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 14,
                  }
    plt.legend(loc='lower right', framealpha=1.0, prop=font_label)
    # 设置标题
    font_axis = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 25,
                 }
    plt.xlabel('Loaction error threshold', font_axis)
    plt.ylabel('precision', font_axis)

    plt.title('OPE Precision Plots on CMOTB-Test', fontsize=19)

    plt.savefig(r'/home/zxl/results/tomp101_satsot/precession-rate.png')

#pr percison of rate 两个框中心距离与阈值的图像

threshold = list(np.arange(0, 51, 1))
# 真值文件夹路径

gtpath = r'/home/zxl/datasets/satsot/GT' #标注好的groundtruth
# 需要画图的文件夹路径
#bbpath = r'/home/zxl/PycharmProjects/pytracking/pytracking/tracking_results/tomp'
bbpath =r'/home/zxl/datasets/satsot/bb2'     #跟踪结果得到的bounding box
algo_accuracy, key_list = calculate_accuracy(threshold, bbpath, gtpath)
# print("key_list",key_list)
plot_figure(threshold, algo_accuracy, sorted(key_list, reverse=True))

