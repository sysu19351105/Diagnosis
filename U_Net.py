# -- coding: utf-8 --

import  torch
import  torch.nn as nn
import  torch.nn.functional as F

class cnnblock(nn.Module):
    def __init__(self,in_channle,out_channle):
        super(cnnblock,self).__init__()
        self.cnn_conv1=nn.Conv2d(in_channle,out_channle,3,1,1)
        self.bn1=nn.BatchNorm2d(out_channle)
        self.ac1=nn.LeakyReLU()

        self.cnn_conv2=nn.Conv2d(out_channle,out_channle,3,1,1)
        self.bn2=nn.BatchNorm2d(out_channle)
        self.ac2=nn.LeakyReLU()
    
    def forward(self,x):
        x=self.cnn_conv1(x)
        x=self.bn1(x)
        x=self.ac1(x)
        x=self.cnn_conv2(x)
        x=self.bn2(x)
        x=self.ac2(x)
        return x

class Upsample(nn.Module):#使用上采样后插值，参数较少
    def __init__(self,c):
        super(Upsample,self).__init__()
        self.conv1=nn.Conv2d(c,c//2,1,1)
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        x=self.conv1(x)
        return x

class Atantion(nn.Module):#注意力机制网�?
    def __init__(self,inc,outc):
        super(Atantion,self).__init__()
        self.down8xconv=nn.Conv2d(inc,11,8,8)
        self.conv1=nn.Conv2d(11,11,11,1,5)
        self.ac1=nn.LeakyReLU()
        self.conv2=nn.Conv2d(11,11,11,1,5)#此处选用11*11卷积核，目的使最终感受野能达到全局
        self.ac2=nn.LeakyReLU()
        self.conv3=nn.Conv2d(11,11,11,1,5)
        self.ac3=nn.LeakyReLU()
        self.up8xconv=nn.ConvTranspose2d(11,outc,8,8)
    def forward(self,x):
        x=self.down8xconv(x)
        x=self.conv1(x)
        x=self.ac1(x)
        x=self.conv2(x)
        x=self.ac2(x)
        x=self.conv3(x)
        x=self.ac3(x)
        x=self.up8xconv(x)
        return x

# class U_Net(nn.Module):
#     def __init__(self):
#         super(U_Net,self).__init__()
#         self.block1=cnnblock(1,64)
#         self.maxpool=nn.MaxPool2d(2)
#         self.block2=cnnblock(64,128)        
#         self.block3=cnnblock(128,256)
#         self.block4=cnnblock(256,512)
#         self.block5=cnnblock(512,1024)
#         self.up1=nn.ConvTranspose2d(1024,512,2,2)
#         self.block6=cnnblock(1024,512)
#         self.up2=nn.ConvTranspose2d(512,256,2,2)
#         self.block7=cnnblock(512,256)
#         self.up3=nn.ConvTranspose2d(256,128,2,2)
#         self.block8=cnnblock(256,128)
#         self.up4=nn.ConvTranspose2d(128,64,2,2)
#         self.block9=cnnblock(128,64)
#         self.finalconv=nn.Conv2d(64,11,1,1,0)
#         self.finalac=nn.LeakyReLU()
#         self.atantion=Atantion(11,11)

#     def forward(self,x):
#         out1=self.block1(x)
#         out2=self.block2(self.maxpool(out1))
#         out3=self.block3(self.maxpool(out2))
#         out4=self.block4(self.maxpool(out3))
#         out5=self.block5(self.maxpool(out4))
#         in6=torch.cat([self.up1(out5),out4],1)
#         out6=self.block6(in6)
#         in7=torch.cat([self.up2(out6),out3],1)
#         out7=self.block7(in7)
#         in8=torch.cat([self.up3(out7),out2],1)
#         out8=self.block8(in8)
#         in9=torch.cat([self.up4(out8),out1],1)
#         out9=self.block9(in9)
#         predict_1=self.finalac(self.finalconv(out9))
#         predict_2=self.atantion(predict_1)
#         predict=predict_1*predict_2
#         return predict_1,predict_2,predict

class U_Net(nn.Module):
    def __init__(self):
        super(U_Net,self).__init__()
        self.block1=cnnblock(1,64)
        self.maxpool=nn.MaxPool2d(2)
        self.block2=cnnblock(64,128)        
        self.block3=cnnblock(128,256)
        self.block4=cnnblock(256,512)
        self.block5=cnnblock(512,1024)
        self.up1=nn.ConvTranspose2d(1024,512,2,2)
        # self.up1=Upsample(1024)
        self.block6=cnnblock(1024,512)
        self.up2=nn.ConvTranspose2d(512,256,2,2)
        # self.up2=Upsample(512)
        self.block7=cnnblock(512,256)
        self.up3=nn.ConvTranspose2d(256,128,2,2)
        # self.up3=Upsample(256)
        self.block8=cnnblock(256,128)
        self.up4=nn.ConvTranspose2d(128,64,2,2)
        # self.up4=Upsample(128)
        self.block9=cnnblock(128,64)
        self.finalconv=nn.Conv2d(64,11,1,1,0)
        self.finalac=nn.ReLU()
        self.atantion=Atantion(11,11)

    def forward(self,x):
        out1=self.block1(x)
        out2=self.block2(self.maxpool(out1))
        out3=self.block3(self.maxpool(out2))
        out4=self.block4(self.maxpool(out3))
        out5=self.block5(self.maxpool(out4))
        in6=torch.cat([self.up1(out5),out4],1)
        out6=self.block6(in6)
        in7=torch.cat([self.up2(out6),out3],1)
        out7=self.block7(in7)
        in8=torch.cat([self.up3(out7),out2],1)
        out8=self.block8(in8)
        in9=torch.cat([self.up4(out8),out1],1)
        out9=self.block9(in9)
        predict=self.finalac(self.finalconv(out9))
        return predict

if __name__ == '__main__':
    a = torch.randn(2, 1, 256, 256)
    net = U_Net()
    # _,_,p=net(a)
    p=net(a)
    print(p.shape)  # torch.Size([2, 11, 256, 256])

        



