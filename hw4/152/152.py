# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mimg
import scipy.ndimage as img
import util

try:
    matfile
except:
    dino1_color, dino_2_color, pts1, pts2 = util.load_dino('dino2.mat')
    dino1= np.dot(dino1_color, [0.299, 0.587, 0.114])
    dino2= np.dot(dino_2_color, [0.299, 0.587, 0.114])

#def gaussian(I,sigma):
#    w = 3*2*sigma+1
#    z = 3*sigma + 1
#    kernel = np.empty([w,w])
#    for i in range(w):
#        for j in range(w):
#            kernel[i,j] = np.exp(-((i-z+1)**2+(j-z+1)**2)/2/sigma**2)
#    kernel /= sum(sum(kernel))
#    return img.filters.convolve(I,kernel)
#
#try:
  dino1_sigma
except:
  dino1_sigma = [None]*6
  dino1_sigma[1] = gaussian(dino1,1)
  dino1_sigma[3] = gaussian(dino1,3)
  dino1_sigma[5] = gaussian(dino1,5)

try:
    dino1_gradient
except:
#    dino1_gradient = [None]*6
#    ix2_element = [None]*6
#    iy2_element = [None]*6
#    ixiy_element = [None]*6
#    for i in (1,3,5):
#        dino1_gradient[i] = np.gradient(dino1_sigma[i])
#        ix2_element[i] = np.square(dino1_gradient[i][1])
#        iy2_element[i] = np.square(dino1_gradient[i][0])
#        ixiy_element[i] = np.multiply(dino1_gradient[i][0],dino1_gradient[i][1])
#    print("pre-calculation stage 1 done")

def score(sigma):
    w = 3*2*sigma+1
    ix2 = img.filters.convolve(ix2_element[sigma], np.ones((w,w)))
    iy2 = img.filters.convolve(iy2_element[sigma], np.ones((w,w)))
    ixiy= img.filters.convolve(ixiy_element[sigma], np.ones((w,w)))
    print("pre-calculation stage 2 done")
    dino1_score = np.empty_like(dino1) 
    for i, row in enumerate(dino1_score):
        for j, item in enumerate(row):
            dino1_score[i,j] = min(np.linalg.eigvals(np.array([[ix2[i,j],ixiy[i,j]],[ixiy[i,j],iy2[i,j]]])))
        if not i%100:
            print(i/len(dino1_score))
    return dino1_score

try:
    dino1_score
except:
    dino1_score = [None]*6
    dino1_score[1] = score(1)
    dino1_score[3] = score(3)
    dino1_score[5] = score(5)
    print("score computed")
    
try: 
    dino1_score_mask
except:
    dino1_score_mask = [None]*6
    dino1_score_mask[1] = np.zeros_like(dino1)
    dino1_score_mask[3] = np.zeros_like(dino1)
    dino1_score_mask[5] = np.zeros_like(dino1)
    for i in range(1,len(dino1)-1):
        for j in range(1,len(dino1[0])-1):
            for sigma in (1,3,5):
                if dino1_score[sigma][i,j] >= np.amax(dino1_score[sigma][i-1:i+2,j-1:j+2].flatten()):
                    dino1_score_mask[sigma][i,j] = 1
        if not i%100:
            print(i/len(dino1))

try:
    pts
except:
    pts = list()
    for i in range(len(dino1_score_mask[5])):
        for j in range(len(dino1_score_mask[5][0])):
            if dino1_score_mask[5][i,j]:
                pts.append([dino1_score[5][i,j],(j,i)])
    pts.sort(key=lambda x:x[0])
    
util.draw_points(dino1_color, [sublist[1] for sublist in pts[-50:]])               
