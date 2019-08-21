"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
import pandas as pd
from torch.autograd import Variable as V
import cv2
import numpy as np
import random
from osgeo import gdalnumeric
import os

def load_img(path):
    img = gdalnumeric.LoadFile(path)
    #img = np.transpose(img,[1,2,0])
    img = np.array(img, dtype="float")
    '''B, G, R = cv2.split(img)
    B = (B - np.mean(B))
    G = (G - np.mean(G))
    R = (R - np.mean(R))
    img_new = cv2.merge([B, G, R])'''
    img_new = img / 255.0
    return img_new

def default_loader(filename,root1,root2,root3,root4,root5):
    #print(filename)
    img1=load_img(root1 + '/' + filename)
    img2=load_img(root2 + '/' + filename)
    mask = gdalnumeric.LoadFile(root3 + '/' +filename).astype(np.float32)
    mask_branch1 = gdalnumeric.LoadFile(root4 + '/' + filename).astype(np.float32)
    mask_branch2 = gdalnumeric.LoadFile(root5 + '/' + filename).astype(np.float32)
    #print(mask[mask > 0])
    #img = np.concatenate([img1,img2,img1 - img2],axis=-1)
    mask[mask > 0] = 1
    mask_branch2[mask_branch2 > 0] = 1
    mask_branch1[mask_branch1 > 0] = 1
    '''rot_p = random.random()
    flip_p = random.random()
    if (rot_p < 0.5):
        pass
    elif (rot_p >= 0.5):
        for k in range(3):
            img1[k, :, :] = np.rot90(img1[k, :, :])
            img2[k, :, :] = np.rot90(img2[k, :, :])
        mask = np.rot90(mask)
        mask_branch1 = np.rot90(mask_branch1)
        mask_branch2 = np.rot90(mask_branch2)

    if (flip_p < 0.25):
        pass
    elif (flip_p < 0.5):
        for k in range(3):
            img1[k, :, :] = np.fliplr(img1[k, :, :])
            img2[k, :, :] = np.fliplr(img2[k, :, :])
        mask = np.fliplr(mask)
        mask_branch1 = np.fliplr(mask_branch1)
        mask_branch2 = np.fliplr(mask_branch2)
    elif (flip_p < 0.75):
        for k in range(3):
            img1[k, :, :] = np.flipud(img1[k, :, :])
            img2[k, :, :] = np.flipud(img2[k, :, :])
        mask = np.flipud(mask)
        mask_branch1 = np.flipud(mask_branch1)
        mask_branch2 = np.flipud(mask_branch2)
    elif (flip_p < 1.0):
        for k in range(3):
            img1[k, :, :] = np.fliplr(np.flipud(img1[k, :, :]))
            img2[k, :, :] = np.fliplr(np.flipud(img2[k, :, :]))
        mask = np.fliplr(np.flipud(mask))
        mask_branch1 =np.fliplr( np.flipud(mask_branch1))
        mask_branch2 = np.fliplr(np.flipud(mask_branch2))
    #img=np.transpose(img,[2,0,1])'''
    mask=np.expand_dims(mask,axis=0)
    mask_branch1 = np.expand_dims(mask_branch1, axis=0)
    mask_branch2 = np.expand_dims(mask_branch2, axis=0)
    return  img1,img2,mask,mask_branch1,mask_branch2

def default_loader_val(filename,root1,root2,root3,root4,root5):
    #print(filename)
    img1=load_img(root1 + '/' + filename)
    img2=load_img(root2 + '/' + filename)
    mask = gdalnumeric.LoadFile(root3 + '/' +filename).astype(np.float32)
    mask_branch1 = gdalnumeric.LoadFile(root4 + '/' + filename).astype(np.float32)
    mask_branch2 = gdalnumeric.LoadFile(root5 + '/' + filename).astype(np.float32)
    #print(mask[mask > 0])
    #img = np.concatenate([img1,img2,img1 - img2],axis=-1)
    mask[mask > 0] = 1
    mask_branch2[mask_branch2 > 0] = 1
    mask_branch1[mask_branch1 > 0] = 1
    #img=np.transpose(img,[2,0,1])
    mask=np.expand_dims(mask,axis=0)
    mask_branch1 = np.expand_dims(mask_branch1, axis=0)
    mask_branch2 = np.expand_dims(mask_branch2, axis=0)
    return  img1,img2,mask,mask_branch1,mask_branch2






class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root1,root2,root3,root4,root5):
        table = pd.read_table(trainlist, header=None, sep=',')
        trainlist = table.values
        self.ids = trainlist
        self.loader = default_loader
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        self.root4 = root4
        self.root5 = root5




    def __getitem__(self, index):
        id = self.ids[index][0]
        img1,img2, mask,mask_branch1,mask_branch2 = self.loader(id, self.root1,self.root2,self.root3,self.root4,self.root5)
        img1 = torch.Tensor(img1.copy())
        img2 = torch.Tensor(img2.copy())
        mask = torch.Tensor(mask.copy())
        mask_branch2 = torch.Tensor(mask_branch2.copy())
        mask_branch1 = torch.Tensor(mask_branch1.copy())
        return img1,img2,mask,mask_branch1,mask_branch2

    def __len__(self):
        return len(self.ids)

class ImageFolder_val(data.Dataset):

    def __init__(self, trainlist, root1,root2,root3,root4,root5):
        table = pd.read_table(trainlist, header=None, sep=',')
        trainlist = table.values
        self.ids = trainlist
        self.loader = default_loader_val
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        self.root4 = root4
        self.root5 = root5




    def __getitem__(self, index):
        id = self.ids[index][0]
        img1,img2, mask,mask_branch1,mask_branch2 = self.loader(id, self.root1,self.root2,self.root3,self.root4,self.root5)
        img1 = torch.Tensor(img1.copy())
        img2 = torch.Tensor(img2.copy())
        mask = torch.Tensor(mask.copy())
        mask_branch2 = torch.Tensor(mask_branch2.copy())
        mask_branch1 = torch.Tensor(mask_branch1.copy())
        return img1,img2,mask,mask_branch1,mask_branch2

    def __len__(self):
        return len(self.ids)