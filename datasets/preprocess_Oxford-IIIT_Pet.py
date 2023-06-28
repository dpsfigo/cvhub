#!usr/bin/env python
# encoding:utf-8
from __future__ import division
 
 
"""
功能： 数据处理模块
"""
 
 
import os
import shutil
 
 
def splitImg2Category(dataDir="./data/oxford-IIIT_Pet/images/",resDir="./data/oxford-IIIT_Pet/preprocessed/"):
    '''
    归类图像到不同目录中
    '''
    for one_pic in os.listdir(dataDir):
        one_path=dataDir+one_pic
        oneDir=resDir+one_pic.split('_')[0].strip()+"/"
        if not os.path.exists(oneDir):
            os.makedirs(oneDir)
        shutil.copy(one_path,oneDir+one_pic)

if __name__ == "__main__":
    splitImg2Category()