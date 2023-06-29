<!--
 * @Author: dpsfigo
 * @Date: 2023-06-27 15:40:01
 * @LastEditors: dpsfigo
 * @LastEditTime: 2023-06-29 15:33:03
 * @Description: 请填写简介
-->
# cvhub

记录分类和实例分割实现
数据集使用Oxford-IIIT Pet Dataset
## 1 Oxford-IIIT Pet数据集介绍
1, Class ID 是对应于pet_label_map.pbtxt的ID值 
2, SPECIES是总分类:1:猫 2:狗 
3, BREED ID :在分类下面的子分类序号,对于总分类1猫其序号为1-25;对于总分类2狗，其序号为1-12。