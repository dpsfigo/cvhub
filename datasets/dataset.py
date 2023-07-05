'''
Author: dpsfigo
Date: 2023-06-29 17:46:45
LastEditors: dpsfigo
LastEditTime: 2023-07-05 14:46:40
Description: 请填写简介
'''
import os
import numpy as np
import cv2
import torch

def get_img_list(data_root, filename):
    data = np.loadtxt(os.path.join(data_root, filename), dtype="str")
    name = data[:,0]
    label = data[:,2].astype(int)-1
    filelist = np.column_stack((name, label))
    return filelist
    

class Dataset():
    def __init__(self, data_root, filelist_root, filelist) -> None:
        self.data = get_img_list(filelist_root, filelist)
        self.data_root = data_root
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        while 1:
            data = self.data[index]
            img = cv2.imread(os.path.join(self.data_root,data[0]+".jpg"))
            img = cv2.resize(img, (224, 224))
            label = int(data[1])
            # x = torch.FloatTensor(img)
            # y = torch.FloatTensor(label)
            return img, label