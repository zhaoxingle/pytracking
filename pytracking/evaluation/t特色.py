import os
sequence_info_list = []
import cv2

image=cv2.imread(r'/home/zxl/datasets/VISO-dataset/sot/ship/045/img/0001.jpg')
# list=[]
for root,dirs,files in os.walk(r"/home/zxl/datasets/VISO-dataset/sot"):
        if root.split('/')[-1]=='gt':
            temp={"name": "null", "path": "null", "startFrame": 'null', "endFrame": 'null', "nz": 4, "ext": "jpg",
             "anno_path": "null","object_class": "null"}

            name=root.split('/')[-3]
            number=root.split('/')[-2]
            path=os.path.join('/'+name,number,"img") #/ship/045/img

            for i in range(len(files)):
                if files[i].split("_")[0]=='1':
                    #print(files[i])
                    startFrame=files[i].split("_")[1]
                    endFrame = files[i].split("_")[2].split(".")[0]
                    temp['startFrame'] = int(startFrame)
                    temp['endFrame'] = int(endFrame)
                    anno_path = os.path.join('/' + name, number, "gt", files[i])  # /ship/045/img
                    temp['anno_path'] = anno_path
            temp['name']=name
            temp['path']=path
            temp['object_class'] = name
            print(name+number)
            list.append(name+number+"\n")
print(list)
f = open(r'/home/zxl/datasets/VISO-dataset/list2.txt', 'w')
f.writelines(list)
f.close()

            #print(temp)
            #sequence_info_list.append(temp)
            # print(root)
            # print(dirs)
            # print(files)
            #print("=------------------------")
#print(sequence_info_list)