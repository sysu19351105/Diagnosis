# -- coding: utf-8 --
import ast
import numpy as np
name_dic = {'T12-L1': 0, 'L1-L2': 1, 'L2-L3': 2, 'L3-L4': 3,
            'L4-L5': 4, 'L5-S1': 5, 'L1': 6, 'L2': 7, 'L3': 8, 'L4': 9, 'L5': 10}

genre_dic = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 4}  # 用于2,5分类标签


def load_all(path):
    # 加载所有信息，以x,y,genre_disc,genre_ver方式(numpy矩阵类型)返回
    # 其中genre_disc[0]代表T12-L1的类�?genre_ver[0]代表'L1'的类�?
    genre_disc = np.zeros((6, 5))  # 6种非L型分类标�?
    genre_ver = np.zeros((5, 2))  # 5种L型分类标�?
    x = np.zeros(11, dtype='int')
    y = np.zeros(11, dtype='int')
    f = open(path, mode='r')
    for line in f.readlines():
        pos1 = line.find(',')
        pos2 = line.find(',', pos1+1)
        other = line[pos2+1:]
        other_dic = ast.literal_eval(other)
        iden = name_dic[other_dic['identification']]

        if iden < 6:  # 为非L型切�?分类情况可能不只分到一�?

            genre_str = other_dic['disc']  # 得到'v2,v5'类型的字符串
            genre_list = genre_str.split(',')  # 得到['v2','v5']类型的list
            for i in genre_list:
                j = genre_dic[i]
                genre_disc[iden, j] = 1
        else:  # 为L型切�?
            j = genre_dic[other_dic['vertebra']]
            genre_ver[iden-6, j] = 1
        x[iden] = int(line[0:pos1])
        y[iden] = int(line[pos1+1:pos2])
    f.close()
    return x, y, genre_disc, genre_ver


def load_location(path):
    # 加载位置信息，以x,y矩阵方式返回
    x = np.zeros(11, dtype='int')
    y = np.zeros(11, dtype='int')
    f = open(path,mode= 'r')
    for line in f.readlines():
        pos1 = line.find(',')
        pos2 = line.find(',', pos1+1)
        other = line[pos2+1:]
        other_dic = ast.literal_eval(other)
        iden = name_dic[other_dic['identification']]
        x[iden] = int(line[0:pos1])
        y[iden] = int(line[pos1+1:pos2])
    return x, y




