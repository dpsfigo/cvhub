'''
Author: dpsfigo
Date: 2023-06-29 17:46:45
LastEditors: dpsfigo
LastEditTime: 2023-07-04 19:47:56
Description: 请填写简介
'''
import os
import numpy as np
import cv2

def get_img_list(data_root, filename):
    data = np.loadtxt(os.path.join(data_root, filename), dtype="str")
    name = data[:,0]
    label = data[:,2].astype(int)
    filelist = np.column_stack((name, label))
    return filelist
    

class Dataset():
    def __init__(self, data_root, filelist) -> None:
        self.data = get_img_list(data_root, filelist)
        self.data_root = data_root
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        filename = self.data[index]
        img = cv2.imread(filename)
        label = self.label[index]
        return img, label
        pass
        