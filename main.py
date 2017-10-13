import numpy as np 
import os
from glob import glob
from scipy import misc
from define_parameters import paths
import re
import random
#from training import perform_training
#from testing import perform_testing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(imagelist):
    # Read Data
    X_img = [misc.imread(file) for file in imagelist]
    X = np.array(X_img)
    return X

def label_encoding(imagelist):
    # Use Regular Expression to get the name of the Data folder
    y_img = []
    for image in imagelist:
        count_slash = image.count('/')
        pattern=""
        for i in range(count_slash-1):
            pattern=pattern+".*/"
        pattern=pattern+"(.*?)_"
        y_img.append(re.search(pattern, image).group(1))
    # Recode labelling
    if(len(set(y_img)) == 2):
        le = preprocessing.LabelEncoder()
        le.fit(np.array(list(set(y_img))))
        y = le.transform(np.array(y_img))
    else:
        logger.info("Hot encoder implemented yet. Exit.")
        exit()
    return y

def shuffle_array(X,y):
	# Shuffle the two categories within X and y consistently
    index = random.sample(range(0,len(y)), len(y))
    X = X[index]
    y = y[index]
    return X,y

def TT_split(X,y):
    # Split training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    return X_train, X_test, y_train, y_test

#def train(X_train, y_train):
    # Perform training

#def test(X_test, y_test):
    # Perform testing

def main():
    # Main calling all functions
	imagelist = glob(paths.DATA_PATH+"*")
	X = read_data(imagelist)
	y = label_encoding(imagelist)
	X, y= shuffle_array(X, y)
	X_train, X_test, y_train, y_test = TT_split(X,y)
	#model = train(X_train, y_train)
	#performance = test(X_test, y_test)
	#return performance

if __name__ == '__main__':
	main()

