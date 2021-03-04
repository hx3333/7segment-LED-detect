# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:04:11 2020

@author: sshss
"""


from __future__ import division
import time, cv2 ,os
import torch 
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
import os.path as osp


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")
    return names

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='FOR semi font OCR')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "images", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "ocr_result", type = str)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.2)
    parser.add_argument("--mdpath", dest = "mdpath", help = "Trained models for classifying the font", default = "resnet18_v1.pth",type=str)
    return parser.parse_args()

    
def get_test_input(input_dim, CUDA):
    img = cv2.imread("testone.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()

    return img_
    
def detect(image_tensor):
    args = arg_parse()
    CUDA = torch.cuda.is_available()
    path = args.mdpath
    images = args.images
    thresh = float(args.confidence)
    inp_dim = 32
    num_classes = 11
    classes = load_classes('classes.txt')
    model = torch.load(path)
    # out = model(get_test_input(inp_dim, CUDA))        #test
    out = model(image_tensor)
    
    # pred & get the biggest confidence class
    _, indices = torch.sort(out, descending=True)
    idx = indices[0][0]
    # return confidence
    percentage = nn.functional.softmax(out, dim=1)[0]
    confidence = '%.4f'%percentage[idx].item()
    
    return classes[idx], confidence
    
if __name__ ==  '__main__':
    detect()
    