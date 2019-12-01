from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()
	def extract_color_histogram(image, bins=(8, 8, 8)):
	
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	else:
		cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()
	# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="../input/plantvillage/")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())
