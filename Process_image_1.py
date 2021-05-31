import cv2
import os
import numpy as np
import cut8
from glob import glob
from skimage.transform import resize

genre_dic = {0: 'v1', 1: 'v2', 2: 'v3', 3: 'v4', 4: 'v5'}
def process(load_path, save_path1, save_path2):


    # 处理训练集数据
    img_cut, genre_disc, genre_ver = cut8.cut_o(path_data=load_path)
    n1 = np.zeros(5)
    n2 = np.zeros(2)
    for i in range(img_cut.shape[0]):

        for j in range(6):

            # label = np.nonzero(genre_disc[i][j])
            if genre_disc[i][j][4] == 1:
                cv2.imwrite(save_path1 + "v5/" + str(int(n1[4])) + '.jpg', img_cut[i][j] * 255)
                n1[4] += 1
            else:
                for n in range(4):
                    if genre_disc[i][j][n] == 1:
                        a = n
                cv2.imwrite(save_path1 + genre_dic[a] + '/' + str(int(n1[a])) + '.jpg', img_cut[i][j] * 255)
                n1[a] += 1
        for j in range(5):

            if genre_ver[i][j][0] == 0:
                cv2.imwrite(save_path2 + "v2/" + str(int(n2[1])) + '.jpg', img_cut[i][j + 6] * 255)
                n2[1] += 1
            else:
                cv2.imwrite(save_path2 + "v1/" + str(int(n2[0])) + '.jpg', img_cut[i][j + 6] * 255)
                n2[0] += 1


def process_test(images, x, y):
    num_ver = 0
    num_disc = 0
    imgpath_list = glob(r'C:\Users\86180\PycharmProjects\test\data' + '/*.jpg')
    imgpath_list.sort()
    infpath_list = glob(r'C:\Users\86180\PycharmProjects\test\data'+ '/*.txt')
    infpath_list.sort()
    img_, genre_disc, genre_ver = cut8.cut_o(path_data=r'C:\Users\86180\PycharmProjects\test\data')  # 测试集存放路径

    img_cut = cut8.cut_u(images, x, y)  # 其中x:(testsize,11) y:(testsize,11) img:(51,256,256)
    '''
    for i in range(img_cut.shape[0]):
        for j in range(img_cut.shape[1]):
            if j % 2 == 0:
                cv2.imwrite(save_path3 + str(num_disc) + '.jpg', img_cut[i][j] * 255)
                num_disc = num_disc + 1
                print("has written into path3")
            else:
                cv2.imwrite(save_path4 + str(num_ver) + '.jpg', img_cut[i][j] * 255)
                print("has written into path4")
                num_ver = num_ver + 1
    '''
    n1 = np.zeros(5)
    n2 = np.zeros(2)
    for i in range(img_cut.shape[0]):

        for j in range(6):

            # label = np.nonzero(genre_disc[i][j])
            if genre_disc[i][j][4] == 1:
                cv2.imwrite(save_path3 + "v5/" + str(int(n1[4])) + '.jpg', img_cut[i][j] * 255)
                print("has written into path3")
                n1[4] += 1
            else:
                for n in range(4):
                    if genre_disc[i][j][n] == 1:
                        a = n
                cv2.imwrite(save_path3 + genre_dic[a] + '/' + str(int(n1[a])) + '.jpg', img_cut[i][j] * 255)
                print("has written into path3")
                n1[a] += 1
        for j in range(5):

            if genre_ver[i][j][0] == 0:
                cv2.imwrite(save_path4 + "v2/" + str(int(n2[1])) + '.jpg', img_cut[i][j + 6] * 255)
                n2[1] += 1
            else:
                cv2.imwrite(save_path4 + "v1/" + str(int(n2[0])) + '.jpg', img_cut[i][j + 6] * 255)
                n2[0] += 1


train_path = "D:/PyCharm/ResNet/homework/train/data"
test_path = "D:/PyCharm/ResNet/homework/test/data"
save_path1 = "D:/PyCharm/ResNet/disc_data/train/"
save_path2 = "D:/PyCharm/ResNet/ver_data/train/"
save_path3 = "C:/Users/86180/PycharmProjects/ResNet/disc_data/valid/"  # 可以修改 D:/PyCharm/ResNet/disc_data/test/
save_path4 = "C:/Users/86180/PycharmProjects/ResNet/ver_data/valid/"  # 可以修改
