# -- coding: utf-8 --
import torch
from glob import glob
import cv2 as cv
import numpy as np
from tools import load_location
def translate(image, x, y): 
    M = np.float32([[1, 0, x], [0, 1, y]]) 
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0])) 
    return shifted 
def rotate(image,x,y,degree):
    x0=image.shape[1]/2
    y0=image.shape[0]/2
    M=cv.getRotationMatrix2D((y0,x0),degree,1.0)
    shifted=cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    theta=np.arctan((y-y0)/(x-x0+0.0001))
    d=np.sqrt(np.square(x-x0)+np.square(y-y0))
    x=np.round(x0+np.cos(theta-degree*np.pi/180)*d)
    y=np.round(y0+np.sin(theta-degree*np.pi/180)*d)
    return shifted,x,y

class firststage_Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.imgpath_list = glob(path+'/*.jpg')
        self.imgpath_list.sort()
        self.infpath_list = glob(path+'/*.txt')
        self.infpath_list.sort()
        self.size=256

    def __getitem__(self, index):
        img = cv.imread(self.imgpath_list[index])
        img = img[:, :, 0]
        h,w=img.shape
        img=cv.resize(img,(self.size,self.size))
        img=img/255.0
        img = torch.from_numpy(img).float()
        img=img.unsqueeze(dim=0)
        x, y = load_location(self.infpath_list[index])
        x=np.round(x*self.size/w)
        y=np.round(y*self.size/h)
        x=torch.from_numpy(x)
        y=torch.from_numpy(y)
        return img,x,y
    
    def __len__(self):
        return len(self.imgpath_list)

class firststage_Datasetplus(torch.utils.data.Dataset):
    def __init__(self, path):
        self.imgpath_list = glob(path+'/*.jpg')
        self.imgpath_list.sort()
        self.infpath_list = glob(path+'/*.txt')
        self.infpath_list.sort()
        self.size=256

    def __getitem__(self, index):
        img = cv.imread(self.imgpath_list[index])
        img = img[:, :, 0]
        h,w=img.shape
        img=cv.resize(img,(self.size,self.size))
        img=img/255.0
        x, y = load_location(self.infpath_list[index])
        x=np.round(x*self.size/w)
        y=np.round(y*self.size/h)

        p1=np.random.random()
        if p1<0.3:
            dx=np.random.randint(-5, 5)
            dy=np.random.randint(-5, 5)
            img=translate(img,dx,dy)
            x=x+dx
            y=y+dy
        p2=np.random.random()
        if p2<0.3:
            gamma=np.random.uniform(0.5, 2.0)
            gamma=np.round(gamma,2)
            img=np.power(img,gamma)
        p3=np.random.random()
        # if p3<0.3:
        #     degree=np.random.uniform(-10,10)
        #     img,x,y=rotate(img,x,y,degree)
        img = torch.from_numpy(img).float()
        img=img.unsqueeze(dim=0)
        x=torch.from_numpy(x)
        y=torch.from_numpy(y)
        return img,x,y
    
    def __len__(self):
        return len(self.imgpath_list)