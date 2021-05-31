import torch
from torchvision import models, datasets
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader

import os
import cv2
import torch.optim as optim
import torch.nn as nn

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到256*256
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 按概率水平旋转
        transforms.CenterCrop(size=224),  # 中心裁剪到224*224，符合resnet的输入要求
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # 转化为tensor，并归一化
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

dataset = 'D:/PyCharm/ResNet/ver_data'  # 数据集路径
train_directory = os.path.join(dataset, 'train')
test_directory = os.path.join(dataset, 'test')

batch_size = 32
num_classes = 2

data = {  # 读取数据集
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

train_data_size = len(data['train'])
test_data_size = len(data['test'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)  # DataLoader加载数据
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

print(train_data_size, test_data_size)  # 打印训练集和测试集的大小

resnet18 = models.resnet18(pretrained=False)  # 加载ResNet18的预训练模型

for param in resnet18.parameters():
    param.requires_grad = True

fc_inputs = resnet18.fc.in_features
resnet18.fc = nn.Sequential(  # 更改网络结构，适应我们自己的分类需要
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1)
)

resnet18 = resnet18.to('cuda:0')

Loss_ = nn.CrossEntropyLoss()  # 定义交叉熵损失
optimizer_ = optim.Adam(resnet18.parameters())  # 定义Adam优化器


def train(model, Loss, optimizer, epoches=50):  # 训练函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 调用GPU
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epoches):  # 进入epoch循环
        epoch_start = time.time()
        print("Epoch:{}/{}".format(epoch + 1, epoches))
        model.train()

        train_loss = 0.0  # 记录训练误差及准确率信息
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):  # 训练部分
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # 前向传播
            loss = Loss(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = Loss(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                test_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])

        if best_acc < avg_test_acc:
            best_acc = avg_test_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tTesting: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for test : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        # torch.save(model, 'models/' + dataset + '_model_' + str(epoch + 1) + '.pt')
    return model


def valid(x, y, net_path, criterion):  # 验证函数
    net = torch.load(net_path)
    y_pred = net(x)
    loss = criterion(y_pred, y)
    print("验证完成，损失为{}", format(loss.item()))
    return 0


train(resnet18, Loss_, optimizer_, 100)
