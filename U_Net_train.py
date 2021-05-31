# -- coding: utf-8 --
import os
import torch
from torch.nn import init
import torch.nn as nn
import random
import numpy as np
import argparse
import math
from tqdm import tqdm
import cv2 as cv
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
os.environ['QTQPAPLATFORM']='offscreen'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from firststage_Dataset import firststage_Dataset,firststage_Datasetplus
from U_Net import U_Net
# from UNet_3Plus import UNet_3Plus as U_Net

#get heatmap
def get_heatmap(x0,y0,heatmap_size=256,A=1.0,sigmas=3):
    #生成以x0，y0为中心，半径为sigmas，heatmap_size大小的高斯分布热力图
    #使用矩阵运算的方式生成heatmap，大大加快速度
    aranges=torch.arange(heatmap_size)
    #获取坐标网格
    y,x=torch.meshgrid(aranges,aranges)
    x=x.cuda()
    y=y.cuda()
    p,c=x0.shape
    x0=x0.unsqueeze(2)
    x0=x0.unsqueeze(3)
    x0=x0.expand(p,c,heatmap_size,heatmap_size).cuda()
    y0=y0.unsqueeze(2)
    y0=y0.unsqueeze(3)
    y0=y0.expand(p,c,heatmap_size,heatmap_size).cuda()

    sigmas=torch.ones((p,c,heatmap_size,heatmap_size))*sigmas
    sigmas=sigmas.cuda()
    squr_distance=torch.pow(x-x0,2)+torch.pow(y-y0,2)    #计算各坐标与指定坐标的距离平方
    heatmap=torch.exp(-squr_distance/(2*torch.pow(sigmas,2))).float()    #生成以x0，y0为中心，半径为sigmas，heatmap_size大小的高斯分布热力图
    return heatmap

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_device
    best_acc=0.6
    U_net=U_Net()
    for params in U_net.parameters():
        init.normal_(params, mean=0.0, std=0.1)
    U_net=nn.DataParallel(U_net).cuda()

    if config.load_pretrain == True:
        U_net.load_state_dict(torch.load(config.pretrain_dir))  
    # train_dataset = firststage_Dataset(config.traindata_path)
    train_dataset = firststage_Datasetplus(config.traindata_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    test_dataset=firststage_Dataset(config.testdata_path)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=config.test_batch_size)
    #loss
    loss = nn.MSELoss().cuda()
    # Trainloss=nn.BCELoss().cuda()

    #optimizer
    optimizer = torch.optim.Adam(
        U_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    #training
    U_net.train()

    train_acc_each=[]
    train_acc16_each=[]
    train_epoch_each=[]
    test_acc_each=[]
    test_acc16_each=[]
    test_epoch_each=[]

    print("<==Training starts==>")
    t0=time.time()
    sigmas=48
    for epoch in range(config.num_epochs):
        #控制heatmap高斯分布的Sigmas半径
        if epoch==100:
            sigmas=36
        elif epoch==200:
            sigmas=24
        elif epoch==300:
            sigmas=18
        elif epoch==400:
            sigmas=12
        elif epoch==500:
            sigmas=9
        elif epoch==600:
            sigmas=6
        elif epoch==700:
            sigmas=3
            
        for train_image, train_x,train_y in tqdm(train_loader,'Epoch'+str(epoch)):            
            train_image = train_image.cuda()
            train_x=train_x.cuda()
            train_y=train_y.cuda()
            p,c,h,w=train_image.shape
            target=get_heatmap(train_x,train_y,sigmas=sigmas)
            pred=U_net(train_image)
            train_loss=loss(pred,target)         
            optimizer.zero_grad()  
            train_loss.backward()  
            optimizer.step()
        
        train_total,train_right,train_right16=0,0,0
        pre_x=torch.zeros((p,11)).cuda()
        pre_y=torch.zeros((p,11)).cuda()
        train_loss_list=[]  
        for i in range (p):
            for j in range(11):
                ind_max=torch.argmax(pred[i,j,:,:])
                pre_y[i,j]=(ind_max/w).floor()
                pre_x[i,j]=ind_max-pre_y[i,j]*w
                train_mseloss=loss(pre_x[i,j],train_x[i,j])+loss(pre_y[i,j],train_y[i,j])
                train_total=train_total+1
                train_loss_list.append(train_mseloss.item())
                if (train_mseloss<9):
                    train_right=train_right+1
                if(train_mseloss<16):
                    train_right16=train_right16+1
        print('Epoch{}: train_loss:{} train_locationloss:{} train_acc:{} train_acc16:{}'.format(epoch,train_loss,np.mean(train_loss_list),train_right/train_total,train_right16/train_total))
        train_acc_each.append(train_right/train_total)
        train_acc16_each.append(train_right16/train_total)
        train_epoch_each.append(epoch)

        if ((epoch+1)%10==0):
            testloss_list=[]
            U_net.eval()
            test_total,test_right,test_right16=0,0,0
            for test_image, test_x,test_y in tqdm(test_loader):
                test_image=test_image.cuda()
                test_x=test_x.cuda()
                test_y=test_y.cuda()       
                pred=U_net(test_image)
                p,_,h,w=pred.shape
                pre_x=torch.zeros((p,11)).cuda()
                pre_y=torch.zeros((p,11)).cuda()

                for i in range (p):
                    for j in range(11):
                        ind_max=torch.argmax(pred[i,j,:,:])
                        pre_y[i,j]=(ind_max/w).floor()
                        pre_x[i,j]=ind_max-pre_y[i,j]*w
                        test_mseloss=loss(pre_x[i,j],test_x[i,j])+loss(pre_y[i,j],test_y[i,j])
                        testloss_list.append(test_mseloss.item())
                        test_total=test_total+1
                        if (test_mseloss<9):
                            test_right=test_right+1
                        if(test_mseloss<16):
                            test_right16=test_right16+1
            print('Epoch{}: testlocation_loss:{}  test_acc:{} test_acc16:{}'.format(epoch,np.mean(testloss_list),test_right/test_total,test_right16/test_total))
            test_acc_each.append(test_right/test_total)
            test_acc16_each.append(test_right16/test_total)
            test_epoch_each.append(epoch)
            if ((test_right/test_total)>best_acc):
                torch.save(U_net.state_dict(),config.run_result_folder+'best.pth')
                print('The best model haved been saved.')
                best_acc=test_right/test_total

            
            if ((epoch+1)%100==0):
                for i in range(11):
                    cv.imwrite(config.run_result_folder+'Epoch'+str(epoch+1)+'_pred_'+str(i)+'.jpg',pred[0,i,:,:].cpu().detach().numpy()*255)
                cv.imwrite(config.run_result_folder+'Epoch'+str(epoch+1)+'_pred.jpg',pred[0,:,:,:].cpu().detach().numpy().sum(axis=0)*255)

            U_net.train()
        torch.save(U_net.state_dict(),config.run_result_folder+'last.pth')

    #plot
    acc_result=plt.figure()
    plt.title('Acc of each epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(train_epoch_each,train_acc_each,color='r',label='Train')
    plt.plot(test_epoch_each,test_acc_each,color='b',label='Test')
    plt.legend()
    acc_result.savefig(config.run_result_folder+'Acc_result.png')

    acc16_result=plt.figure()
    plt.title('Acc of each epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(train_epoch_each,train_acc16_each,color='r',label='Train')
    plt.plot(test_epoch_each,test_acc16_each,color='b',label='Test')
    plt.legend()
    acc16_result.savefig(config.run_result_folder+'Acc16_result.png')
    print('Training complete.')
    print('{} epochs completed in {} hours.'.format(config.num_epochs, (time.time() - t0) / 3600))

    
        



if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    startepoch = 0
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--GPU_device', type=str,default='4,5')#GPU使用序列，可以为'0'或'0,1,2,3'
    parser.add_argument('--traindata_path', type=str,default='/data/L_E_Data/Diagnosis_data/train/data') #训练数据路径
    parser.add_argument('--testdata_path', type=str,default='/data/L_E_Data/Diagnosis_data/test/data')    #测试数据路径
    parser.add_argument('--lr', type=float,default=0.0001)  
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=4)  
    parser.add_argument('--run_result_folder', type=str,default='/data/L_E_Data/Diagnosis_data/run/trainSigmoid/')#训练结果输出路径
    parser.add_argument('--load_pretrain', type=bool,default=False)
    parser.add_argument('--pretrain_dir', type=str,default='/data/L_E_Data/Diagnosis_data/run/trainsigma3dataplus/best.pth')#预训练模型路径
    config = parser.parse_args()

    if not os.path.exists(config.run_result_folder):
        os.mkdir(config.run_result_folder)

    train(config)
