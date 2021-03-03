# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:45:31 2020

@author: sshss
"""

import numpy as np
import cv2, os, random,glob
import matplotlib.pyplot as plt

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def displayimg(img, winname):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)


def plot_hist(img):
    chans = cv2.split(img)
    colors = ("b","g","r")
    plt.figure()
    plt.xlim([0,256])
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    for (color,chan) in zip(colors,chans):
        hist = cv2.calcHist([chan], [0], None, [256], [0,256])
        plt.plot(hist,color=color)
        
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

def calcsobel(gray, ksize, weightX, weightY):
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize = ksize)    #sobel edge detect
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize = ksize)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX,weightX,absY,weightY,0)
    return sobel





























