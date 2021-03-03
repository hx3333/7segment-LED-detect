# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:25:50 2020

@author: sshss
"""

import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt


def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img

def gamma(image,coef):
    fimg = image/255.0
    dst = np.power(fimg,coef)*255
    dst = np.round(dst)
    dst = dst.astype('uint8')
    return dst

def normalize(image):
    Imax = np.max(image)
    Imin = np.min(image)
    Omin,Omax = 10,155
    a = float(Omax-Omin)/(Imax-Imin)
    b = Omin - a*Imin
    dst = a*image + b
    dst = dst.astype("uint8")
    return dst

def calcsobel(img, ksize, weightX, weightY):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=ksize)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=ksize)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX,weightX,absY,weightY,0)
    return sobel

def unevenLightCompensate(gray, blockSize, lightcomp_kernel, lightcomp_sigma):
    average = np.mean(gray)   
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]
            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype("uint8")
    dst = cv2.GaussianBlur(dst, lightcomp_kernel, lightcomp_sigma)
    return dst
    



def main(filenames):
    for filename in filenames:
        glassid = filename.split('_')[-5]
        required_char = ['D','E','F','4','G','H','I','J','K','L','M','N','O','P','Q','R','S','U','V','W','X','Y','Z']
        temp_list = [i in glassid for i in required_char]
        if True in temp_list:
            img = cv2.imread(filename,0)
            img_x,img_y = img.shape[:2]
            
            ################rotate the image####################
            
            if (img_x < img_y):
                img = img[::-1,::-1]
            else:
                img = RotateClockWise90(img)
                
            hist = cv2.calcHist([img], [0], None, [256], [0,255])
            maxLoc = np.where(hist==np.max(hist))
            grayscale = maxLoc[0][0]
            
            if grayscale > 90:
                coef = 3
            else:
                coef =0.33
            
            copied = np.copy(img)
            copied = cv2.cvtColor(copied, cv2.COLOR_GRAY2BGR)
            gamma_changed = gamma(img, coef)
            normalized = normalize(gamma_changed)
            
            ###################################################
            
            ############## Image processing ###################
            flatten_img =  unevenLightCompensate(normalized, 50, (3,3), 5)    #flatten intensity
            sobel = calcsobel(flatten_img, 3, 0.5, 0.5)     #sobel gradient
            th,dst = cv2.threshold(sobel,0,255,cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dst = cv2.medianBlur(dst, 3)
            dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel,iterations=1)
            
            image, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i,c in enumerate(contours):
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(copied, (x,y), (x+w,y+h), (0,0,255))
                if (35<w<45) and (85<h<95):
                    char = img[y:y+h,x:x+w]
                    cv2.imwrite(glassid+str(i)+'.jpg',char)
        else:
            continue
            
if __name__ == '__main__':
    filenames = glob.glob('models/32inch/*.jpg')
    main(filenames)










