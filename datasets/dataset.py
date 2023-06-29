'''
Author: dpsfigo
Date: 2023-06-29 17:46:45
LastEditors: dpsfigo
LastEditTime: 2023-06-29 19:37:19
Description: 请填写简介
'''
import numpy as np

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
        
        pass
        