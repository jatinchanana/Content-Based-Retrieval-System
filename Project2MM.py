#!/usr/bin/env python
# coding: utf-8

import cv2
import pickle
from sklearn.cluster import MiniBatchKMeans
from skimage import color
import numpy as np
from PIL import Image
import glob
extractor=cv2.xfeatures2d.SURF_create(1500,2,1,1,0) # SURF feature extractor

'''For each image we create a final feature vector,using build_histogram method, which is a histogram of visual words that each image has'''
def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(2000)
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram
def gray(img):
    return(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY))

folders = glob.glob(r'\path\to\dataset\paris*\*')
image_list = []
for folder in folders:
    for f in glob.glob(folder+'\*.jpg'):
        im=Image.open(f)
        image_list.append(im)
        
def extract_features(image):
    kp,desc = extractor.detectAndCompute(image,None)
    return desc

'''Extracting SURF features for each image in the dataset and appending it to descriptor list'''
descriptor_list=[]
for i,image in enumerate(image_list):
    if i%100==0:
        print(str(i)+"images done")
    image = gray(image)
    descriptor = extract_features(image)
    if (descriptor is not None):
        descriptor_list.extend(descriptor)
'''Training a MiniBatch kmeans algorithm on the descriptor list'''
kmeans = MiniBatchKMeans(n_clusters=2000,random_state=0,batch_size=300000,max_iter=10).fit(descriptor_list)

preprocessed_image = []
hists = []
folders = glob.glob(r'C:\Users\jatin\Desktop\paris*\*')
image_list = []
for folder in folders:
    for f in glob.glob(folder+'\*.jpg'):
        im=Image.open(f)
        grayimage=gray(im)
        descriptor =  extract_features(grayimage)
        if(descriptor is not None):
            histogram= build_histogram(descriptor,kmeans)
            preprocessed_image.append([f[23:],histogram]) # Store [imagename,feature vector] for all images. 
            hists.append(histogram) # build final feature vector to fit Nearest Neighbors algorithm 

'''Using nearest neighbors to find 10 closest images to the query image'''
from sklearn.neighbors import NearestNeighbors

data = cv2.imread(r'\path\to\the\query\image')
data = gray(data)
descriptor = extract_features(data)
histogram = build_histogram(descriptor, kmeans)
neighbor = NearestNeighbors(n_neighbors =10,metric='cosine') 
neighbor.fit(hists)
dist, result = neighbor.kneighbors([histogram])  #result returns the index of the 10 closest images returned
