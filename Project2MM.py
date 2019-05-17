#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pickle
from sklearn.cluster import MiniBatchKMeans
from skimage import color
import numpy as np
from PIL import Image
import glob
extractor=cv2.xfeatures2d.SURF_create(1500,2,1,1,0)


# In[2]:


def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(2000)
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram
def gray(img):
    return(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY))


# In[ ]:


folders = glob.glob(r'C:\Users\jatin\Desktop\paris*\*')
image_list = []
for folder in folders:
    for f in glob.glob(folder+'\*.jpg'):
        im=Image.open(f)
        image_list.append(im)


# In[20]:


def extract_features(image):
    kp,desc = extractor.detectAndCompute(image,None)
    return desc


# In[ ]:


'''Extracting SURF features for each image in the dataset'''
descriptor_list=[]
for i,image in enumerate(image_list):
    if i%100==0:
        print(str(i)+"images done")
    image = gray(image)
    descriptor = extract_features(image)
    if (descriptor is not None):
        descriptor_list.extend(descriptor)


# In[ ]:


kmeans = MiniBatchKMeans(n_clusters=2000,random_state=0,batch_size=300000,max_iter=10).fit(descriptor_list)


# In[ ]:


with open('my_kmeans_ORB_classifier.pkl', 'wb') as fid:
    pickle.dump(kmeans, fid)


# In[4]:


with open('my_kmeansfinalcheck_classifier.pkl', 'rb') as fid:
    kmeans = pickle.load(fid)


# In[21]:


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
            preprocessed_image.append([f[23:],histogram])
            hists.append(histogram)  


# In[ ]:


from sklearn.neighbors import NearestNeighbors

data = cv2.imread(r'\path\to\the\query\image')
data = gray(data)
descriptor = extract_features(data)
histogram = build_histogram(descriptor, kmeans)
neighbor = NearestNeighbors(n_neighbors =10,metric='cosine')
neighbor.fit(hists)
dist, result = neighbor.kneighbors([histogram])

