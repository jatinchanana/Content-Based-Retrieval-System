from flask import Flask, render_template, url_for, request
from flask_bootstrap  import Bootstrap
from sklearn.externals import joblib
import pandas as pd
#from get_imgs import get_imgs
import os
import cv2
import pickle
from sklearn.cluster import MiniBatchKMeans
from skimage import color
import numpy as np
from PIL import Image
import glob
import time
extractor = cv2.xfeatures2d.SURF_create(1500,2,1,1,0)#cv2.xfeatures2d.SIFT_create()
from sklearn.neighbors import NearestNeighbors
def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(2000) #2000  
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram
def gray(img):
    return(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY))
def extract_features(image):
    kps,desc=extractor.detectAndCompute(image,None)
    return desc
with open(os.path.abspath('models/my_kmeansfinalcheck_classifier.pkl'), 'rb') as fid:
    kmeans = pickle.load(fid)
with open(os.path.abspath('models/All_imagedata.pkl'),'rb') as fid:
    preprocessed_image = pickle.load(fid)
with open(os.path.abspath('models/my_Allimagefeatures.pkl'), 'rb') as fid:
   hists = pickle.load(fid)

app = Flask(__name__)

Bootstrap(app) 

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
@app.route('/')

def index():
	return render_template('upload.html')

@app.route('/upload',methods=['POST'])

def upload():
	start = time.time()
	target = os.path.join(APP_ROOT,'query/')
	if not os.path.isdir(target):
		os.mkdir(target)
	for file in request.files.getlist("file"):
		filename="query.jpg" #file.filename
		temp = "/".join([target,filename])
		file.save(temp)
	data = cv2.imread(os.path.abspath('query')+'/'+filename)
	data = gray(data)
	descriptor = extract_features(data)
	histogram = build_histogram(descriptor, kmeans)
	neighbor = NearestNeighbors(n_neighbors =10,metric='cosine')
	neighbor.fit(hists)
	dist, result = neighbor.kneighbors([histogram])
	image_name_list = []
	result_folder = os.path.join(APP_ROOT,'static/')
	if not os.path.isdir(result_folder):
			os.mkdir(result_folder)
	for i,file in enumerate(result[0]):
		file1 = Image.open(os.path.join(os.path.abspath(""),preprocessed_image[file][0]))
		filename='result'+str(i)+str(time.time())+'.jpg'
		image_name_list.append(filename)
		temp = "/".join([result_folder,filename])
		file1.save(temp)
		#image_name_list.append(file1)
	totaltime = round((time.time() - start),2)
	return render_template('results.html',pred_imgs=image_name_list, queryimg = image_name_list[0],runtime=totaltime)
if __name__ == '__main__':
	app.run(debug=True)