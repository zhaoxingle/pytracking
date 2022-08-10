# 只需修改路径 line349,line351

import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps
from numpy import trapz
import os
import glob
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib as mpl

import pdb

fm = mpl.font_manager

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']


def get_box_pos(GT_path, BB_path):
    '''
    读取文件内部的数据，其格式为left_x left_y width height 以\t分割
    参数为groundtruth_rect,boundbox单个文件路径
    返回两个文件内部的数据列表
    '''
    gt_box = []  # 存放groundtruth_rect中的数据
    bb_box = []  # 存放bound_box中的数据
    with open(GT_path) as gf:  # 获取groundtruth_rect中box的位置(left_top_x,left_top_y,width,height)
        # lines = gf.readlines()
        # lines = np.loadtxt(GT_path,delimiter=',')
        # lines = np.loadtxt(GT_path,delimiter='\t')

        try:
            lines = np.loadtxt(GT_path, delimiter=',')
        except:
            lines = np.loadtxt(GT_path)

        for line in lines:
            gt_box.append(line)

    with open(BB_path) as bf:  # 获取bound_box中box的位置(left_top_x, left_top_y, width, height)

        try:
            lines = np.loadtxt(BB_path, delimiter=',')
        # else:
        except:
            lines = np.loadtxt(BB_path)
        # print("lines",lines)
        for line in lines:
            bb_box.append(line)

    return gt_box, bb_box


# 两个box是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
def bb_overlab(box1, box2):
    '''
    说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
    x: 框的左上角 x 坐标
    y: 框的左上角 y 坐标
    w: 检测框的宽度
    h: 检测框的高度
    :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    '''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if (x1 > x2 + w2):
        return 0
    if (y1 > y2 + h2):
        return 0
    if (x1 + w1 < x2):
        return 0
    if (y1 + h1 < y2):
        return 0
    colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)


def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def get_SR(threshold, gt_path, bb_path):
    '''
    计算success-rate 参数分别是阈值列表，人工标注数据目录，算法跟踪数据目录
    返回值:success-rate为字典，键是面积+算法名，值是列表，里面存放算法名，面积，sr
    key_list为为列表，存放字典里面的键，用于排序，便于设置图例顺序
    '''
    # gt_path下是真实数据,均为文件,bb_path_dir下各种算法进行目标跟踪的数据,有多个目录
    bb_dir = []
    success_rate = {}
    algorithm_name = []
    key_list = []
    # dell=['7.txt','114.txt','115.txt','12.txt','129.txt','131.txt','142.txt','145.txt','176.txt','197.txt','320.txt','335.txt','394.txt']
    dell = []
    # 获取bb_path下面的多个算法的目录路径
    # for root, dirs, files in os.walk(bb_path):
    #     for dire in dirs:
    #         bb_dir.append(os.path.join(root, dire))
    #         algorithm_name.append(dire)
    for name in os.listdir(bb_path):
        # 确保是算法名
        if os.path.isdir(os.path.join(bb_path, name)):
            algorithm_name.append(name)
            bb_dir.append(os.path.join(bb_path, name))

    gt_file_list = sorted(os.listdir(gt_path))  # 存放groundtruth_rect文件的路径

    index = 0
    # 分别遍历bb_path下的各个算法目录中的文件，并计算对应的sr
    for dirpath in bb_dir:
        algorithm_info = []
        IOU_list = []  # 存放该跟踪算法重叠度
        single_SR = []  # 存放该跟踪算法的在不同阈值下的success_rate

        # 获取一个跟踪算法下文件的路径
        track_files = sorted(os.listdir(dirpath))

        for fpath in gt_file_list:  # 遍历该跟踪算法下的数据文件
            # 计算box的position
            # print("processing {} ,{}".format(dirpath.split('/')[-1],fpath))
            gt_fpath = os.path.join(gt_path, fpath)
            bb_fpath = os.path.join(dirpath, fpath)
            # pdb.set_trace()
            gt_box_list, bb_box_list = get_box_pos(gt_fpath, bb_fpath)

            # 计算重叠度IOU
            # print(len(gt_box_list), len(bb_box_list))
            for j in range(len(bb_box_list)):  # 遍历文件内数据

                IOU = rect_iou(gt_box_list[j], bb_box_list[j])

                IOU_list.append(IOU)

        # 计算sr
        length = len(IOU_list)
        for ts in threshold:
            cnt = 0
            for iou in IOU_list:
                if iou > ts:
                    cnt = cnt + 1
            single_SR.append(cnt / length)

        # 计算面积
        x = np.array(threshold)
        y = np.array(single_SR)
        area = trapz(y, x, dx=0.001)
        area = '%.03f' % area
        # area=y[25]
        # area = str('%.03f' % area)
        algorithm_info.append(algorithm_name[index])
        algorithm_info.append(area)
        algorithm_info.append(single_SR)

        key = '[' + area + ']' + algorithm_name[index]
        # pdb.set_trace()

        key_list.append(key)
        success_rate[key] = algorithm_info
        index = index + 1

    return success_rate, key_list


def SR_plot(threshold, success_rate, key_list):  # ,list_name,k
    # 设置横纵坐标的刻度间距
    plt.figure(figsize=(10, 10))
    plt.xticks(list(np.arange(0, 1.1, 0.1)), ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
    plt.yticks(list(np.arange(0, 1.0, 0.1)), ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    # plt.yticks(list(np.arange(0,1.1,0.1)), ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])
    # 设置坐标轴刻度字体大小
    plt.tick_params(labelsize=13)
    plt.grid(alpha=1, ls='--')
    plt.axis([0, 1, 0, 0.9])  # 设置坐标轴起始点
    colors = ['red', 'darkred', 'yellow', 'dodgerblue', 'black', 'lime', 'darkorchid', 'cyan', 'slategrey', 'maroon',
              'rosybrown', \
              'deeppink', 'coral', 'tan', 'green', 'magenta', 'pink', 'olive', 'gold', 'plum', 'peru', 'chocolate',
              'crimson', \
              'crimson', 'deepskyblue', 'springgreen', 'slategrey', 'plum', 'steelblue', 'lawngreen', 'royalblue']
    # s
    # colors = [ 'darkred','red', 'dodgerblue','yellow',  'black','darkorchid','rosybrown','lime', 'slategrey','cyan','maroon',\
    #    'magenta', 'coral',  'tan','pink','deeppink','green',  'olive', 'gold','plum','peru','chocolate','crimson',\
    #        'crimson', 'deepskyblue',  'springgreen', 'slategrey', 'plum', 'steelblue', 'lawngreen','royalblue']
    # y
    # colors = [ 'red','darkred',  'dodgerblue', 'black','yellow', 'darkorchid','lime','rosybrown','slategrey','cyan','maroon',\
    #    'magenta', 'coral', 'tan','deeppink', 'pink', 'green', 'olive', 'gold','plum','peru','chocolate','crimson',\
    #        'crimson', 'deepskyblue',  'springgreen', 'slategrey', 'plum', 'steelblue', 'lawngreen','royalblue']
    linestyles = ['solid', '--', '-', 'solid', '-.']
    i = 0
    x = np.array(threshold)
    t = len(key_list)
    if t % 2 == 1:
        t = (t + 1) / 2
    else:
        t = t / 2
    for key in key_list:
        print(key)
        info = success_rate[key]
        name = info[0]
        area = info[1]
        sr = info[2]
        y = np.array(sr)
        # 对图像进行拟合成光滑的曲线
        x_smooth = np.linspace(x.max(), x.min())
        y_smooth = make_interp_spline(x, y)(x_smooth)
        if i < t:
            plt.plot(x_smooth, y_smooth, color=colors[i], label=key, linewidth=6, linestyle='-')
        else:
            plt.plot(x_smooth, y_smooth, color=colors[i], label=key, linewidth=6, linestyle='--')
        i = i + 1

    # framealpha设置图例背景的透明度

    font_label = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 14,
                  }
    # plt.legend(loc='lower left',framealpha=1.0,prop=font_label,labels="f")
    plt.legend(loc='lower left', framealpha=1.0, prop=font_label)

    font_axis = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 25,
                 }
    # plt.rcParams['font.sans-serif']=['SimHei']
    # matplotlib.rcParams['axes.unicode_minus']=False
    # plt.xlabel('Overlap threshold',fontsize='19')
    # plt.ylabel('Success rate', fontsize='19')
    plt.xlabel('Overlap threshold', font_axis)
    plt.ylabel('Success rate', font_axis)
    # plt.title('OPE Success Plots on CMOTB-Test', fontsize=19)
    # plt.savefig(r"C:\Users\Wyl\Desktop\89\MarMOT-yan\success-rate(y).png") # 保存图片
    # plt.savefig(r"C:\Users\Wyl\Desktop\89\MarMOT-yan\success-rate(y).eps") # 保存图片

    # plt.title('OPE Success Plots on CMOTB-'+ list_name +'('+ str(k) +')', fontsize=19)
    # plt.savefig(r"C:\Users\Wyl\Desktop\M\success-rate("+ list_name +").png") # 保存图片
    # plt.savefig(r"C:\Users\Wyl\Desktop\M\success-rate("+ list_name +").eps") # 保存图片

    # plt.title('OPE Success Plots on Mode Switching Times ('+ list_name+'('+ str(k) +')', fontsize=19)#切换次数
    # plt.title('OPE Success Plots on Switching Times ('+ list_name +')'+'('+ str(k) +')', fontsize=19)#切换次数
    # plt.savefig(r"C:\Users\Wyl\Desktop\模态切换\success-rate("+ list_name +").png") # 保存图片
    # plt.savefig(r"C:\Users\Wyl\Desktop\模态切换\success-rate("+ list_name +").eps") # 保存图片

    plt.title('OPE Success Plots on CMOTB-Test', fontsize=19)
    plt.savefig(r"/home/zxl/results/tomp101_satsot/success-rate.png")  # 保存图片
    # plt.savefig(r"/home/user/zhutianhao/success-rate.eps") # 保存图片

    # plt.show()


'''
sttr = ['216-1','216-2','216-3','216-4']
#sttr=["SV","ARC","FM","OV","WFA","MB","BC","SOB","IPR","POC","FOC","Switch"]
for t in sttr:    
    print(t)
    path_GT = 'C:/Users/Wyl/Desktop/GT-' + t
    path_BD = 'C:/Users/Wyl/Desktop/'+ t 
    threshold = list(np.arange(0,1.01,0.02))
    path_file_number=glob.glob(path_GT +'/*.txt')#或者指定文件下个数    
    k = len(path_file_number)
    sr, key = get_SR(threshold, path_GT, path_BD)
    SR_plot(threshold, sr, sorted(key,reverse=True),t,k)




sttr=["1","2","3","4"]
for t in sttr:    
    print(t)
    path_GT = 'C:/Users/Wyl/Desktop/GT-' + t
    path_BD = 'C:/Users/Wyl/Desktop/'+ t
    threshold = list(np.arange(0,1.01,0.02))
    path_file_number=glob.glob(path_GT +'/*.txt')#或者指定文件下个数    
    k = len(path_file_number)
    sr, key = get_SR(threshold, path_GT, path_BD)
    SR_plot(threshold, sr, sorted(key,reverse=True),t,k)

'''
# sr SR is the percentage of the frames whose overlap ratio
# between the output bounding box and the ground truth bounding box
# is larger than a threshold, and we compute the representative SR score
# by the area under the curves.
path_GT = r'/home/zxl/datasets/satsot/GT'
# 需要画图的文件夹路径
path_BD = r'/home/zxl/datasets/satsot/bb2'
threshold = list(np.arange(0, 1.01, 0.02))
path_file_number = glob.glob(path_GT + '/*.txt')  # 或者指定文件下个数
k = len(path_file_number)
sr, key = get_SR(threshold, path_GT, path_BD)
SR_plot(threshold, sr, sorted(key, reverse=True))




