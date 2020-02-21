from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random

from utils.imutils2 import *
from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc

import cv2


# get the video frames
# two patches in the future frame, one is center, the other is one of the 8 patches around

class VIPSet(data.Dataset):
    def __init__(self, params, is_train=True, resnet=False):

        self.filelist = params['filelist']
        self.batchSize = params['batchSize']
        self.imgSize = params['imgSize']
        self.cropSize = params['cropSize']
        self.cropSize2 = params['cropSize2']
        self.videoLen = params['videoLen']
        self.predFrames = params['predFrames'] # 4
        self.sideEdge = params['sideEdge'] # 64

        self.sampleRate = params['sampleRate']

        self.videojpgs = '/media/xtli/eb0943df-a3fc-4ae2-a6e5-021cfdcfec3d/home/xtli/DATA/VIP/VIP_Videos/video_jpgs/'


        # prediction distance, how many frames far away
        self.predDistance = params['predDistance']
        # offset x,y parameters
        self.offset = params['offset']
        # gridSize = 3
        # self.gridSize = params['gridSize']

        self.is_train = is_train

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []
        self.vidfiles = []
        self.resnet = resnet

        for line in f:
            rows = line.split(' ')
            jpgfile = rows[0]
            lblfile = rows[1].strip()

            foldernames = jpgfile.split('/')
            videoname = foldernames[-1]

            self.vidfiles.append(self.videojpgs + videoname + '/')

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()

    def cropimg(self, img, offset_x, offset_y, cropsize):

        img = im_to_numpy(img)
        cropim = np.zeros([cropsize, cropsize, 3])
        cropim[:, :, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize, :]
        cropim = im_to_torch(cropim)

        return cropim


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]
        vid_path   = self.vidfiles[index]

        imgs = []
        lbls = []
        patches = []
        target_imgs = []

        framelist = os.listdir(folder_path)
        framelist = np.sort(framelist)

        startid = int(framelist[0][:-4])
        endid   = int(framelist[-1][:-4])

        vidlist = []
        for i in range(startid, endid, self.sampleRate):
            vidlist.append("{:012d}.jpg".format(i))

        for i in range(len(framelist)):
            vidlist.append(framelist[i])

        vidlist = np.unique(vidlist)
        vidlist = np.sort(vidlist)

        # print(vidlist)

        frame_num = len(vidlist) + self.videoLen

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        # print(len(vidlist))

        for i in range(frame_num):
            if i < self.videoLen:
                img_path = vid_path + vidlist[0]
                # lbl_path = label_path + "{:05d}.png".format(0)
            else:
                img_path = vid_path + vidlist[i - self.videoLen]
                # lbl_path = label_path + "{:05d}.png".format(i - self.videoLen)

            if(self.resnet):
                img = load_image(img_path)  # CxHxW
            else:
                img = load_image_lab(img_path)
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd

            if ht <= wd:
                ratio  = 1.0 #float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
                newh = self.imgSize
                neww = int(self.imgSize * ratio)
            else:
                ratio  = 1.0 #float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))
                newh = int(self.imgSize * ratio)
                neww = self.imgSize

            if i == 0:
                imgs = torch.Tensor(frame_num, 3, newh, neww)

            img = color_normalize(img, mean, std)
            imgs[i] = img

            # lblimg  = scipy.misc.imread(lbl_path)
            # lblimg  = scipy.misc.imresize( lblimg, (newh, neww), 'nearest' )

            # lbls.append(lblimg.copy())

        gridx = 0
        gridy = 0

        sideEdge = self.sideEdge
        gridy = int(newh / sideEdge)
        gridx = int(neww / sideEdge)

        lbllist = os.listdir(label_path)
        lbllist = np.sort(lbllist)
        lbl_path = label_path + "/" + lbllist[0]

        lblimg  = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        # lblimg  = cv2.resize(lblimg, (neww, newh), interpolation=cv2.INTER_NEAREST)
        # lbls.append(lblimg.copy())



        #
        # for i in range(frame_num):
        #
        #     img = imgs[i]
        #     ht, wd = img.size(1), img.size(2)
        #     newh, neww = ht, wd
        #
        #     sideEdge = self.sideEdge
        #
        #     gridy = int(newh / sideEdge)
        #     gridx = int(neww / sideEdge)
        #
        #     # img = im_to_numpy(img)
        #     # target_imgs.append(img)
        #
        #     for yid in range(gridy):
        #         for xid in range(gridx):
        #
        #             patch_img = img[:, yid * sideEdge: yid * sideEdge + sideEdge, xid * sideEdge: xid * sideEdge + sideEdge].clone()
        #             # patch_img = im_to_torch(patch_img)
        #             # patch_img = resize(patch_img, self.cropSize2, self.cropSize2)
        #             # patch_img = color_normalize(patch_img, mean, std)
        #
        #             patches.append(patch_img)
        #
        #
        # countPatches = frame_num * gridy * gridx
        # patchTensor = torch.Tensor(countPatches, 3, self.cropSize2, self.cropSize2)
        #
        # for i in range(countPatches):
        #     patchTensor[i, :, :, :] = patches[i]


        # patchTensor = patchTensor.view(frame_num, gridy * gridx, 3, self.cropSize2, self.cropSize2)

        print(imgs.size())

        # Meta info
        meta = {'folder_path': folder_path, 'gridx': gridx, 'gridy': gridy}

        # lbls_tensor = torch.Tensor(len(lbls), newh, neww)
        # for i in range(len(lbls)):
        #     lbls_tensor[i] = torch.from_numpy(lbls[i])
        #

        lbls_tensor = torch.from_numpy(lblimg)
        lbls_tensor = lbls_tensor.unsqueeze(0)

        return imgs, lbls_tensor, meta

    def __len__(self):
        return len(self.jpgfiles)
