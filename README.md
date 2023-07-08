<!--
 * @Author: dpsfigo
 * @Date: 2023-06-27 15:40:01
 * @LastEditors: dpsfigo
 * @LastEditTime: 2023-07-08 16:25:52
 * @Description: 请填写简介
-->
# cvhub

记录分类和实例分割实现
数据集使用Oxford-IIIT Pet Dataset和tensorflow flower data
## 1 Oxford-IIIT Pet数据集介绍
1, Class ID 是对应于pet_label_map.pbtxt的ID值。
2, SPECIES是总分类:1:猫 2:狗。
3, BREED ID :在分类下面的子分类序号,对于总分类1猫其序号为1-25;对于总分类2狗，其序号为1-12。

## tensorflow flower data
共5个类别的数据。

## classification说明(以alexnet为例)
每个文件夹下有6个文件：
1，alexnet.py:网络文件。
2，class_indices.json:label文件。
3，finetune.py:以torchvision为基础微调代码。
4，hparams.py:超参文件。
5，preict.py:推理代码
6，train.py:训练代码