import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import rotate, resize
from matplotlib import pyplot as plt
import ast
import os


def find_k(x, y):
    k_all = []
    batch_size = x.shape[0]
    size = x.shape[1]
    for i in range(batch_size):
        f = np.polyfit(x=x[i, :], y=y[i, :], deg=2)
        p = np.poly1d(f)
        p_deriv = p.deriv()
        k_batch = []
        for j in range(size):
            k = p_deriv(x[i, j])
            k_batch.append(k)
        k_all.append(k_batch)
    k_all = np.array(k_all)
    return k_all


def cut(img, k, x, y, l1=144, w1=96, l2=144, w2=96):
    img_cut = []
    batch_size = x.shape[0]
    size = x.shape[1]
    angle = - np.arctan(k) * 180 / 3.1415
    left = x - l1 / 2
    right = x + l1 / 2
    up = y - w1 / 2
    down = y + w1 / 2
    for i in range(batch_size):
        img_b = []
        for j in range(size):
            left[i, j], right[i, j], up[i, j], down[i, j] = assert_lrup(left=left[i, j], right=right[i, j], up=up[i, j],
                                                                        down=down[i, j], img_d=img[i].shape[1],
                                                                        img_r=img[i].shape[0])
            img_o = img[i, int(up[i, j]): int(down[i, j]), int(left[i, j]):int(right[i, j])]
            img_o = rotate(image=img_o, angle=angle[i, j], resize=False)
            img_o = img_o[int(w1*(1/4)): int(w1*(3/4)), int(l1*(1/4)): int(l1*(3/4))]
            img_o = resize(image=img_o, output_shape=(w2, l2))
            img_b.append(img_o)
        img_cut.append(img_b)
    img_cut = np.array(img_cut)
    return img_cut


def assert_lrup(left, right, up, down, img_d, img_r):
    if left < 0:
        left = 0
    if up < 0:
        up = 0
    if right > img_r:
        right = img_r
    if down > img_d:
        down = img_d
    return left, right, up, down


def final_cut(img, x, y, img_shape, l1=144, w1=96, l2=144, w2=96):
    batch_size = img.shape[0]
    for i in range(batch_size):
        img[i] = resize(img[i], (512, 512))
    k = find_k(y, x)
    for i in range(batch_size):
        x[i] = x[i] / img_shape[i, 1] * 512
        y[i] = y[i] / img_shape[i, 0] * 512
    img_cut = cut(img=img, k=k, x=x, y=y, l1=l1, w1=w1, l2=l2, w2=w2)
    return img_cut


name_dic = {'T12-L1': 0, 'L1-L2': 1, 'L2-L3': 2, 'L3-L4': 3, 'L4-L5': 4, 'L5-S1': 5, 'L1': 6, 'L2': 7, 'L3': 8, 'L4': 9, 'L5': 10}

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
        iden = name_dic[other_dic['identification']] #获得name_dic中对应部位的编号
        if iden < 6:  # 为非L型切�?分类情况可能不只分到一�?
            genre_str = other_dic['disc']  # 得到'v2,v5'类型的字符串
            genre_list = genre_str.split(',')  # 得到['v2','v5']类型的list
            for i in genre_list:
                j = genre_dic[i] #获得genre_dic中对应椎间盘的疾病诊断的编号
                genre_disc[iden, j] = 1
        else:  # 为L型切�?
            j = genre_dic[other_dic['vertebra']]
            genre_ver[iden-6, j] = 1 # 'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4
        x[iden] = int(line[0:pos1])
        y[iden] = int(line[pos1+1:pos2])
    f.close()
    return x, y, genre_disc, genre_ver


def load_all_data(path, output_shape):
    x = []
    y = []
    genre_disc = []
    genre_ver = []
    img = []
    index = []
    img_shape = []
    for filename in os.listdir(path=path):
        dot_pos = filename.find('.')
        name = filename[0:dot_pos]
        exten = filename[dot_pos:dot_pos+4]
        if exten == '.jpg':
            img_ = imread(path + '\\' + name + exten)
            img_shape_ = img_.shape
            img_ = resize(image=img_, output_shape=output_shape)
            img.append(img_)
            img_shape.append(img_shape_)
            index_ = name[5:dot_pos]
            index.append(index_)
        elif exten == '.txt':
            x_, y_, genre_disc_, genre_ver_ = load_all(path + '\\' + name + exten)
            x.append(x_)
            y.append(y_)
            genre_disc.append(genre_disc_)
            genre_ver.append(genre_ver_)
    x = np.array(x)
    y = np.array(y)
    genre_disc = np.array(genre_disc)
    genre_ver = np.array(genre_ver)
    img = np.array(img)
    index = np.array(index)
    img_shape = np.array(img_shape)
    return x, y, genre_disc, genre_ver, img, index, img_shape





def cut_o(path_data, output_shape=(512, 512), l1=144, w1=96, l2=144, w2=96):
    """
    cut_o函数用于根据原始坐标数据切分出脊柱块
    参数：
    path_data -- 原始数据路径（包含jpg和txt文件的那个文件夹）
    output_shape -- 读入图像resize大小，默认即可
    l1, w1 -- 用于控制切割方框区域的大小。如果觉得切割方框太小，无法完全罩住脊柱块，可以适当调大l1与w1，同时保持l1与w1的比例为3：2
    l2, w2 -- 输出的图像大小。默认即可，无需改动
    输出：
    img_cut -- 切分好的图像。其中，img_cut.shape=(batch_size, 11, w2, l2)(灰度图)或(batch_size, 11, w2, l2, 3)(rgb图)。这里11个脊柱块的顺序由load_all函数决定，参考load_all函数。
    genre_disc -- 与脊柱块标注相关。这里的genre_disc与load_all函数输出的genre_disc是一致的，参考load_all函数注释。
    genre_ver -- 与脊柱块标注相关。这里的genre_ver与load_all函数输出的genre_ver是一致的，参考load_all函数注释。
    """
    x, y, genre_disc, genre_ver, img, index, img_shape = load_all_data(path=path_data, output_shape=output_shape)
    img_cut = final_cut(img=img, x=x, y=y, img_shape=img_shape, l1=l1, w1=w1, l2=l2, w2=w2)
    return img_cut, genre_disc, genre_ver


def cut_u(img, x, y, l1=144, w1=96, l2=144, w2=96):
    """
    cut_u函数用于根据u_net预测的坐标切分出脊柱块
    参数：
    img -- 原始图像，即使用u_net预测的那一批图像，可以是训练集也可以是测试集，大小为batchsize*512*512或batchsize*512*512*3
    x -- u_net输出结果中的x坐标。其中，x.shape=(batch_size, 11)
    y -- u_net输出结果中的y坐标。其中，y.shape=(batch_size, 11)
    output_shape -- 读入图像resize大小，默认即可
    l1, w1 -- 用于控制切割方框区域的大小。如果觉得切割方框太小，无法完全罩住脊柱块，可以适当调大l1与w1，同时保持l1与w1的比例为3：2
    l2, w2 -- 输出的图像大小。默认即可，无需改动
    输出：
    img_cut -- 切分好的图像。其中，img_cut.shape=(batch_size, 11, w2, l2)(灰度图)或(batch_size, 11, w2, l2, 3)(rgb图)。这里11个脊柱块的顺序由load_all函数决定，参考load_all函数。

    更新：把输入参数改为unet的一个batch，并输入unet预测得到的x和y，作为final_cut的直接输入参数

    """
    # _x, _y, _genre_disc, _genre_ver, img, _index, img_shape = load_all_data(path=path_img, output_shape=output_shape)
    img_cut = final_cut(img=img, x=2*x, y=2*y, img_shape=512*np.ones((51,2)), l1=l1, w1=w1, l2=l2, w2=w2)
    return img_cut

