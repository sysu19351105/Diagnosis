# -- coding: utf-8 --
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time

from U_Net import U_Net
from firststage_Dataset import firststage_Dataset

def test(config):
    testloss_list = []
    U_net=U_Net()
    U_net=nn.DataParallel(U_net).cuda()
    U_net.load_state_dict(torch.load(config.pretrain_dir))
    U_net.eval()
    test_dataset=firststage_Dataset(config.testdata_path)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=config.test_batch_size)

    loss=torch.nn.MSELoss().cuda()
    t0=time.time()
        
        
    test_total,test_right,test_right16=0,0,0
    for test_image, test_x,test_y in tqdm(test_loader):
        test_image=test_image.cuda()
        test_x=test_x.cuda()
        test_y=test_y.cuda()       
        pred=U_net(test_image)
        p,_,h,w=pred.shape
        pre_x=torch.zeros((p,11)).cuda()#保存x坐标
        pre_y=torch.zeros((p,11)).cuda()#保存y坐标

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
    print('testlocation_loss:{}  test_acc:{} test_acc16:{}'.format(np.mean(testloss_list),test_right/test_total,test_right16/test_total))
    t1=time.time()
    print('Speed: {}s/patch.'.format((t1-t0)/(len(test_loader)*config.test_batch_size)))

    if config.iswrite:
        for i in range(11):
            cv.imwrite(config.run_result_folder+'Epoch'+str(epoch+1)+'_pre1_'+str(i)+'.jpg',pred[0,i,:,:].cpu().detach().numpy()*255)
        cv.imwrite(config.run_result_folder+'Epoch'+str(epoch+1)+'_pre1.jpg',pred[0,:,:,:].cpu().detach().numpy().sum(axis=0)*255)





if __name__ == '__main__':
    # 载入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_device', type=str,default='4,5')#GPU使用序列，可以为'0'或'0,1,2,3'
    parser.add_argument('--testdata_path', type=str,default='/data/L_E_Data/Diagnosis_data/test/data')    # 测试数据path
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrain_dir', type=str,default="/data/L_E_Data/Diagnosis_data/run/trainsigma3dataplus/best.pth")  # 加载训练模型的路径?
    parser.add_argument('--iswrite',type=bool,default=False)#是否输出样例图片
    parser.add_argument('--run_result_folder', type=str,default='/data/L_E_Data/Diagnosis_data/run/test/')
    config = parser.parse_args()
    test(config)    
