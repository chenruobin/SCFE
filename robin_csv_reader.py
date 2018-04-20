#封装：CSV数据读取器

import csv
import numpy as np
#输入：csv文件名，标签的列数，特征的个数（从前向后数最后一列是标签，其他的均为特征）
def read_csv_data(filename, num_of_target, num_of_feature):
    birth_data = []
    with open(filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            birth_data.append(row)
    birth_data = [[float(x) for x in row] for row in birth_data]  # 将数据转换为float格式
    #此时，数据依然是list，【】中有‘，’
    y_vals = np.array([x[num_of_target-1] for x in birth_data])
    #np.array将list变为矩阵，方便以后的向量运算
    x_vals = np.array(
        [[x[ix] for ix in range(num_of_feature)] for x in birth_data])
    return x_vals,y_vals


