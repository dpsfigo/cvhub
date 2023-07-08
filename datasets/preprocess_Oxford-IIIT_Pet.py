'''
Author: dpsfigo
Date: 2023-07-08 14:05:46
LastEditors: dpsfigo
LastEditTime: 2023-07-08 16:17:26
Description: 处理oxford-iiit-pet数据，按文件夹保存
'''
import os
from shutil import copy, rmtree
import random
import numpy as np


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 指向你解压后的flower_photos文件夹
    # 读取train，test文件
    data_root = os.path.dirname(os.path.realpath(__file__))
    imgs_path = os.path.join(data_root, "images")
    #把图片归类到不同的目录
    data = {}
    for one_pic in os.listdir(imgs_path):
        one_path=os.path.join(imgs_path,one_pic)
        oneDir=one_pic.split('_')[0].strip()
        if oneDir in data.keys():
            data[oneDir].append(one_path)
        else:
            data[oneDir] = []
            data[oneDir]
            data[oneDir].append(one_path)

    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    
    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for key in data:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, key))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in data:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in data:
        # cla_path = os.path.join(origin_flower_path, cla)
        images = data[key]
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = image
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = image
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()

