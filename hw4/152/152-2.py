# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mimg
import scipy.ndimage as img
import scipy.spatial.distance as dis
import util

try:
    dino2
except:
    dino1_color, dino_2_color, pts1, pts2 = util.load_dino('dino2.mat')
    dino1= np.dot(dino1_color, [0.299, 0.587, 0.114])
    dino2= np.dot(dino_2_color, [0.299, 0.587, 0.114])

def gaussian(I,sigma):
    w = 3*2*sigma+1
    z = 3*sigma + 1
    kernel = np.empty([w,w])
    for i in range(w):
        for j in range(w):
            kernel[i,j] = np.exp(-((i-z+1)**2+(j-z+1)**2)/2/sigma**2)
    kernel /= sum(sum(kernel))
    return img.filters.convolve(I,kernel)

try:
  dino1_sigma
except:
  dino1_sigma = gaussian(dino1,5)

try:
    dino1_gradient
except:
    dino1_gradient = np.gradient(dino1_sigma)
    ix2_element = np.square(dino1_gradient[1])
    iy2_element = np.square(dino1_gradient[0])
    ixiy_element = np.multiply(dino1_gradient[0],dino1_gradient[1])
    
def score():
    w = 3*2*5+1
    ix2 = img.filters.convolve(ix2_element, np.ones((w,w)))
    iy2 = img.filters.convolve(iy2_element, np.ones((w,w)))
    ixiy= img.filters.convolve(ixiy_element, np.ones((w,w)))
    print("pre-calculation done")
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
    dino1_score = score()
    print("score computed")
    
try: 
    dino1_score_mask
except:
    dino1_score_mask = np.zeros_like(dino1)
    for i in range(1,len(dino1)-1):
        for j in range(1,len(dino1[0])-1):
                if dino1_score[i,j] >= np.amax(dino1_score[i-1:i+2,j-1:j+2].flatten()):
                    dino1_score_mask[i,j] = 1
        if not i%100:
            print(i/len(dino1))

try:
    pts
except:
    pts = list()
    for i in range(len(dino1_score_mask)):
        for j in range(len(dino1_score_mask[0])):
            if dino1_score_mask[i,j]:
                pts.append([dino1_score[i,j],(j,i)])
    pts.sort(key=lambda x:x[0])

#for graph 2
try:
  dino2_sigma
except:
  dino2_sigma = gaussian(dino2,5)

try:
    dino2_gradient
except:
    dino2_gradient = np.gradient(dino2_sigma)
    ix2_element = np.square(dino2_gradient[1])
    iy2_element = np.square(dino2_gradient[0])
    ixiy_element = np.multiply(dino2_gradient[0],dino2_gradient[1])
    
def score2():
    w = 3*2*5+1
    ix2 = img.filters.convolve(ix2_element, np.ones((w,w)))
    iy2 = img.filters.convolve(iy2_element, np.ones((w,w)))
    ixiy= img.filters.convolve(ixiy_element, np.ones((w,w)))
    print("pre-calculation done")
    dino2_score = np.empty_like(dino2) 
    for i, row in enumerate(dino2_score):
        for j, item in enumerate(row):
            dino2_score[i,j] = min(np.linalg.eigvals(np.array([[ix2[i,j],ixiy[i,j]],[ixiy[i,j],iy2[i,j]]])))
        if not i%100:
            print(i/len(dino1_score))
    return dino2_score

try:
    dino2_score
except:
    dino2_score = score2()
    print("score computed")
    
try: 
    dino2_score_mask
except:
    dino2_score_mask = np.zeros_like(dino2)
    for i in range(1,len(dino2)-1):
        for j in range(1,len(dino2[0])-1):
                if dino2_score[i,j] >= np.amax(dino2_score[i-1:i+2,j-1:j+2].flatten()):
                    dino2_score_mask[i,j] = 1
        if not i%100:
            print(i/len(dino2))

try:
    pts2
except:
    pts2 = list()
    for i in range(len(dino2_score_mask)):
        for j in range(len(dino2_score_mask[0])):
            if dino2_score_mask[i,j]:
                pts2.append([dino2_score[i,j],(j,i)])
    pts2.sort(key=lambda x:x[0])   

#correspondences

p1 =  pts[-50:]
p2 =  pts2[-50:]

def fNSSD(i,j):
    x,y,a,b = p1[i][1][0], p1[i][1][1], p2[j][1][0], p2[j][1][1]
    u1 = dino1[x-4:x+5,y-4:y+5].mean()
    u2 = dino2[a-4:a+5,b-4:b+5].mean()
    o1 = dino1[x-4:x+5,y-4:y+5].var()
    o2 = dino2[a-4:a+5,b-4:b+5].var()
    P1 = np.copy(dino1_sigma[x-4:x+5,y-4:y+5])
    P2 = np.copy(dino2_sigma[a-4:a+5,b-4:b+5])
    P1 = (P1-u1)/o1
    P2 = (P2-u2)/o2
    try:
        return np.square(np.subtract(P1,P2)).sum()
    except:
        return np.inf

nssd = np.empty([50,50])
ratio = np.empty([50,50])

def rat(t,r):
    cor1 = []
    cor2 = []
    for i in range(50):
        (j1,j2) = np.argsort(nssd[i])[:2]
        if nssd[i,j1] > t:
            continue
        if np.absolute(dis.cdist([p1[i][1]],[p2[j1][1]])/dis.cdist([p1[i][1]],[p2[j2][1]])) < r:
            cor1.append(p1[i][1])
            cor2.append(p2[j1][1])
    return (cor1,cor2)

for i in range(50):
    for j in range(50):
        nssd[i,j] = fNSSD(i,j)
        
cor1,cor2 = rat(300000,1)

util.draw_correspondence(dino1_color, dino_2_color, cor1, cor2)