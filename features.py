import numpy as np
import os
import cv2
import pdb
from scipy.misc import imread

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def DiffI(imgs):
    '''
    input the list of images
    output the It of them on time
    '''
    result = []
    result.append(np.zeros_like(imgs[0]))
    for i in range(len(imgs)-1):
        result.append(imgs[i+1]-imgs[i])
    return result

def Cal_uv(imgs):
    '''
    input the list of images
    output the u,v
    '''
    u = []
    v = []
    for i in range(len(imgs)-1):
        gray0 = imgs[i]
        gray1 = imgs[i+1]

        p0 = np.nonzero(gray0)
        p0 = np.asarray(p0).T
        p0 = np.reshape(p0,[-1,1,2])
        p0 = p0.astype(np.float32)

        p1, _, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk_params)

        diff = np.reshape(p1-p0,[-1,2])

        tempU = np.zeros_like(gray0).astype(np.float32)
        tempV = np.zeros_like(gray0).astype(np.float32)
        for i in range(p0.shape[0]):
            tempU[int(p0[i,0,0]),int(p0[i,0,1])] = diff[i,0]
            tempV[int(p0[i,0,0]),int(p0[i,0,1])] = diff[i,1]

        u.append(tempU)
        v.append(tempV)
    u.append(np.zeros_like(gray0))
    v.append(np.zeros_like(gray0))
    return u,v


filenames = os.listdir("./imagedata")
filenames.sort(key = str.lower)
try:
    p=filenames.index('.DS_Store')
    filenames.pop(p)
except:
    filenames = filenames
print filenames

imgs = []
for filename in filenames:
    imgs.append(np.asarray(imread("./imagedata/"+filename)))

It = DiffI(imgs)
u,v = Cal_uv(imgs)
pdb.set_trace()