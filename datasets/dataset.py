'''
Author: dpsfigo
Date: 2023-06-29 17:46:45
LastEditors: dpsfigo
LastEditTime: 2023-06-30 17:03:16
Description: 请填写简介
'''
import numpy as np
import cv2

def get_img_list(data_root, filename):
    filelist = []
    data = np.loadtxt(filename,dtype="%s")
    name = data_root + data[:,0]
    label = data[:,2]
    return name, label
    

class Dataset():
    def __init__(self, data_root, filelist) -> None:
        self.data, self.label = get_img_list(data_root, filelist)
    
    def __getitem__(self, index):
        filename = self.data[index]
        img = cv2.imread(filename)
        label = self.label[index]
        return img, label
        pass
        