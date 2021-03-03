# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:10:40 2020

@author: sshss
"""

from sklearn.model_selection import train_test_split
import random, os, glob, shutil


def movefile(source, dest):    
    try:
        shutil.move(source, dest)
    except shutil.Error:
        os.remove(source)


new = r'C:\Users\sshss\OCR\datasets_ocr\new_datasets'
tdir = r'C:\Users\sshss\OCR\datasets_ocr\Val'
wdir = r'C:\Users\sshss\OCR\datasets_ocr\Train'


folders = os.listdir(new)
for label in folders:
    char_folder = os.path.join(new,label)
    char_train = os.path.join(wdir,label)
    char_val = os.path.join(tdir,label)
    pics = [os.path.join(char_folder, img) for img in os.listdir(char_folder) if os.path.splitext(img)[1] == '.png' or 
    os.path.splitext(img)[1] =='.jpeg' or 
    os.path.splitext(img)[1] =='.jpg' or 
    os.path.splitext(img)[1] =='.bmp']
    if len(pics)<=1:
        continue
    else:
        train,test = train_test_split(pics, train_size = 0.8)
        for train_img in train:
            movefile(train_img, char_train)
        for test_img in test:
            movefile(test_img, char_val)