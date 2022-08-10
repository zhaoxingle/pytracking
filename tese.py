import os

import torch
#state_dict = torch.load("/home/zxl/PycharmProjects/pytracking/pytracking/networks/tomp101.pth.tar")
#torch.save(state_dict, "/home/zxl/PycharmProjects/pytracking/pytracking/networks/tomp101.pth.tar", _use_new_zipfile_serialization=False)

# import  re
# a=re.search(r'[0-9]+','car028').group()
# b=re.search(r'[a-z]+','car028').group()
# print(a)
# print(b)
import json
f = open(r"/home/zxl/datasets/satsot/SatSOT/SatSOT.json", 'r')
content = f.read()
a = json.loads(content)
f.close()
# for key,values in a.items():
#
#     print(key)
#     print(values)

# import numpy as np
#
# array=np.loadtxt(r'/home/zxl/datasets/1.txt',delimiter=',')
# print(array)

# Python program to explain cv2.imread() method

# importing cv2
# import cv2
#
# # path
# path = r'/home/zxl/datasets/satsot/SatSOT/car_01/img/0001.jpg'
#
# # Using cv2.imread() method
# img = cv2.imread(path)
# print(img)
# Displaying the image
#cv2.imshow('image', img)
import torch
print(torch.cuda.is_available())

# encoding:utf-8

# import cv2
# import numpy as np
# image = cv2.imread(r'/home/zxl/datasets/satsot/SatSOT/train_03/img/0001.jpg')
# GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# x=534
# y=508
# w=322
# h=89

# h, w = image.shape[:2]
# h, w = map(int, [h/4, w/4])
# print(h,w)
# # no flip
# draw_0 = cv2.rectangle(image, (100, 100), (10, 10), (0, 0, 255))#cv2.rectangle(image, pt1,pt2, color)
#x, y, w, h = cv2.boundingRect(GrayImage)
#draw_1 = cv2.rectangle(image, (435, 521), (465,480), (0, 255, 0), 2)
# draw_1 = cv2.rectangle(image, (x, y), (x+w,y+h), (0, 255, 0), 2)
# 参数：pt1,对角坐标１, pt2:对角坐标２
# 注意这里根据两个点pt1,pt2,确定了对角线的位置，进而确定了矩形的位置
# The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners are pt1 and pt2.
#draw_0 = cv2.rectangle(image, (2 * w, 2 * h), (3 * w, 3 * h))

#cv2.imwrite("vertical_flip.jpg", draw_1)  # 将画过矩形框的图片保存到当前文件夹

# cv2.imshow("draw_0", draw_1)  # 显示画过矩形框的图片
# cv2.waitKey(0)
# cv2.destroyWindow("draw_0")

# f = open(r"/home/zxl/datasets/satsot/SatSOT/SatSOT.json", 'r')
# content = f.read()
# dict = json.loads(content)
#
# for k, v in dict.items():
#     #print(v)
#     #print(v['video_dir'])
#     dir_name=v['video_dir']+'.txt'
#     path=os.path.join(r"/home/zxl/datasets/satsot/GT",dir_name)
#     #print(path)
#     gt=v['gt_rect']
#
#     #print(len(gt))
#     for i in range(0,len(gt)):
#         #gt[i]=[1,2,3,4]
#         with open(path,'a') as f:
#             b=''
#             for j in range(0,len(gt[i])):
#
#                 b+=str(gt[i][j])+','
#             #print(b) #450.0,506.0,31.0,51.0,
#             b=b[:-1]+'\n'
#             #b=b.strip(',')
#             #print(b)
#             f.write(b)
#         #print('-------')
#     # with open(path) as f:
#     #     f.writelines()
#     # ground_truth_rect = dict[name]['gt_rect']
#     # ground_truth_rect = np.array(ground_truth_rect)
# f.close()
# with open(r'/home/zxl/PycharmProjects/pytracking/pytracking/tracking_results/tomp/tomp50') as f:
import shutil
import re

root=r'/home/zxl/PycharmProjects/pytracking/pytracking/tracking_results/tomp/tomp50'
for root, dirs, files in os.walk(root):
    for i in range(len(files)):
        file=files[i].split('.')[0]
        if(file.split('_')[-1]!='time'):
            print(files[i])
            a=re.search('[0-9]+',files[i]).group()
            st=str(a)
            if(len(st)==3):
                st=st[1:3]
            print(st)

            path=os.path.join(root,files[i])
            #print(path)
            #画图前要先创建tomp50_000文件夹，不能直接允许。
            path2=r'/home/zxl/PycharmProjects/pytracking/pytracking/tracking_results/tomp/tomp50_000'
            #print(path)
            shutil.move(path,path2)

            #print(files[i])
#
#     #
#     #print(files)
# list=[[1,2],[1,2],[3,3]]
# list=np.array(list)
# list.reshape(len(list),2)
# print(list.shape)
# import io
# def check(_str):
#     if _str[0]==',':
#         _str=_str.strip(',')
#     return _str
# path=r'/home/zxl/datasets/VISO-dataset/sot/car/001/gt/1_1_260(2).txt'
# with open(path, 'r') as f:
#
#     ground_truth_rect = np.loadtxt(io.StringIO(f.read().replace(',', ' ')), converters={0:check})
#     print(ground_truth_rect)
# s = io.StringIO('10.01 31.25-\n19.22 64.31\n17.57- 63.94')
# def ss(fld):
#     return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)
# #np.loadtxt(s, converters=conv)
#
#
# print(np.loadtxt(s, converters=ss))
# from io import StringIO
#
# s = StringIO("0xDE 0xAD\n0xC0 0xDE")
# import functools
# conv = functools.partial(int, base=16)
# np.loadtxt(s, converters=conv)
# print(np.__version__)
# import numpy as np
# X = np.arange(24).reshape((2,3,2,2))
# print(X)
# print('====================')
# print(X[0,:,:,:])
# print('====================')

# print(X[0])
# print (np.mean(X, axis=0, keepdims=True))
# print('================')
# print (np.mean(X, axis=(0,1), keepdims=True))
# print('================')
# print (np.mean(X, axis=1, keepdims=True))
# print('================')
# print (np.mean(X, axis=2, keepdims=True))
