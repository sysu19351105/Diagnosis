import os
import torch
import torch.nn as nn
import numpy as np
import argparse

import time
from skimage.transform import resize

from U_Net import U_Net
from firststage_Dataset import firststage_Dataset
import Process_image_1
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def connect_test(config):
    U_net = U_Net()
    U_net = nn.DataParallel(U_net).cuda()
    U_net.load_state_dict(torch.load(config.pretrain_dir))
    U_net.eval()
    test_dataset = firststage_Dataset(config.testdata_path)
    length = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size)
    t0 = time.time()
    x = torch.zeros((length, 11))
    y = torch.zeros((length, 11))
    k = 0
    imgs = np.zeros(shape=(51, 512, 512))
    num = 0

    for test_image, test_x, test_y in test_loader:

        test_image = test_image.cuda()
        pred = U_net(test_image)
        p, _, h, w = pred.shape
        pre_x = torch.zeros((p, 11)).cuda()  # 保存x坐标
        pre_y = torch.zeros((p, 11)).cuda()  # 保存y坐标
        for i in range(p):
            for j in range(11):
                ind_max = torch.argmax(pred[i, j, :, :])
                pre_y[i, j] = (ind_max / w).floor()
                pre_x[i, j] = ind_max - pre_y[i, j] * w
        x[k:k + p, :] = pre_x
        y[k:k + p, :] = pre_y
        k = k + p

        # ___________________________以下是获取batch图片的代码___________________________________
        for img in test_image:
            img = img.cpu().numpy()
            print("resize前img大小")
            print(img.shape)
            img = resize(image=img, output_shape=(1, 512, 512))  # 改变图像大小为512,512 便于图像切割
            print("resize后img大小")
            print(img.shape)
            imgs[num, :, :] = img
            num = num + 1
    # ____________________________________________________________________________________
    x = x.numpy()
    y = y.numpy()
    print("imgs:")  # imgs:
    print(type(imgs))  # <class 'numpy.ndarray'>
    print(imgs.shape)  # (51, 512, 512)

    return x, y, imgs


transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

Loss_ = nn.CrossEntropyLoss()


if __name__ == '__main__':
    # 载入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_device', type=str, default='0')  # GPU使用序列，可以为'0'或'0,1,2,3'
    parser.add_argument('--testdata_path', type=str, default='./test/data')  # 测试数据path
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrain_dir', type=str, default="./best.pth")  # 加载训练模型的路径?
    config = parser.parse_args()

    x, y, imgs = connect_test(config)  # 获取预测坐标以及对应图片
    Process_image_1.process_test(imgs, x, y)

