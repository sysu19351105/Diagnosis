import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time
from skimage.transform import resize
from torchvision import models, datasets
from cut8 import cut_u
from U_Net import U_Net
from firststage_Dataset import firststage_Dataset
import Process_image_1
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# resnet18 = models.resnet18(pretrained=False)  # 加载ResNet18的预训练模型
#
#
# fc_inputs = resnet18.fc.in_features
# resnet18.fc = nn.Sequential(  # 更改网络结构，适应我们自己的分类需要
#     nn.Linear(fc_inputs, 256),
#     nn.ReLU(),
#     nn.Dropout(0.4),
#     nn.Linear(256, 5),
# )
#
# resnet18 = resnet18.to('cuda:0')


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
    print(imgs.shape)  # (51, 256, 256)

    return x, y, imgs


transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

Loss_ = nn.CrossEntropyLoss()


def valid(net, valid_data):  # 验证函数
    test_loss = 0.0
    test_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.eval()
    for j, (inputs, labels) in enumerate(valid_data):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        loss = Loss_(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        test_acc += acc.item() * inputs.size(0)
    test_data_size = len(valid_data)

    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size
    print("验证完成，损失为:", avg_test_loss, "准确率为：", avg_test_acc)
    return 0


if __name__ == '__main__':
    # 载入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_device', type=str, default='0')  # GPU使用序列，可以为'0'或'0,1,2,3'
    parser.add_argument('--testdata_path', type=str, default='./test/data')  # 测试数据path
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrain_dir', type=str, default="./best.pth")  # 加载训练模型的路径?
    config = parser.parse_args()

    x, y, imgs = connect_test(config)  # 获取256,256大小的图片下对应的预测坐标以及图片
    # Process_image_1.process_test(imgs, x, y)
    valid_path = 'C:/Users/86180/PycharmProjects/ResNet/disc_data/valid'  # 更改路径，以及选择disc/ver
    valid_data = datasets.ImageFolder(root=valid_path, transform=transform)
    valid_data = DataLoader(valid_data, batch_size=32,
                            shuffle=False)
    resnet18 = torch.load('_model_38.pt')
    print(resnet18)
    # resnet18.load_state_dict(net_state)
    valid(resnet18, valid_data)
