'''
Author: dpsfigo
Date: 2023-06-29 17:46:45
LastEditors: dpsfigo
LastEditTime: 2023-06-29 18:00:43
Description: 请填写简介
'''
import numpy as np

def get_img_list(data_root, filename):
    filelist = []
    data = np.loadtxt(filename,dtype="%s")
    

class Dataset():
    def __init__(self, data_root, filelist) -> None:
        self.filelist = filelist
        
    
    def __getitem__(self, index):
        pass
        