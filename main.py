import numpy as np 
import os
from glob import glob
from scipy import misc
from define_parameters import paths, image_parameters
import re
import random
from preprocessing import image_augmentation
from training import train_CNN
from testing import test_CNN
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(imagelist):
    # Read Data
    logger.info(" Reading images data...")
    X_img = [misc.imread(file) for file in imagelist]
    X = np.array(X_img)
    return X

def standardization_data(imagelist, method = 'over-image'):
	# Standardization:  x - mean / std
    logger.info(" Standardization of images...")
    for x in imagelist:
        for chan in range(0,len(x[0,0])):
            x[:,:,chan] = x[:,:,chan]/255
    return imagelist

def label_encoding(imagelist):
    # Use Regular Expression to get the name of the Data folder
    logger.info(" Encoding labels...")
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
    logger.info(" Shuffling data...")
    index = random.sample(range(0,len(y)), len(y))
    X = X[index]
    y = y[index]
    return X,y

def TT_split(X,y):
    # Split training and testing
    logger.info(" Splitting training and testing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
    return X_train, X_test, y_train, y_test

def data_augmentation(X_train, X_test, y_train):
    # Resize both image stats and perform data augmentation on training set
    preprocess = image_augmentation()
    logger.info(" Resize images on testing set...")
    X_reshape = []
    for i in range(0,len(X_test)):
        image = X_test[i] 
        image = preprocess.resize_image(image, npix = image_parameters.SIZE_IMAGE)
        X_reshape.append(image)
    X_test = np.stack(X_reshape)

    logger.info(" Resize and perform data augmentation on training set...")
    X_aug = []
    y_aug = []
    for i in range(0,len(X_train)):
        image = X_train[i]
        images_augmented = preprocess.perform_augmentation(image, npix = image_parameters.SIZE_IMAGE)
        X_aug.extend(images_augmented)
        y_aug += len(images_augmented)* [y_train[i]]
    X_train = np.stack(X_aug)
    y_train = np.stack(y_aug)
    if(len(X_train) != len(y_train) and X_train.ndim != 4): 
        logger.error("  Error in creating augmented arrays. Exit.")
        exit()
    return X_train, X_test, y_train

def train(X_train, y_train):
    # Perform training
    logger.info(" Starting training...")
    process = train_CNN()
    process.main_train(X_train, y_train)

def test(X_test, y_test):
    # Perform testing
    logger.info(" Starting testing...")
    process2 = test_CNN()
    performance = process2.main_test(X_test, y_test)
    return performance

def main():
    # Main calling all functions
    imagelist = glob(paths.DATA_PATH+"*")
    X = read_data(imagelist)
    y = label_encoding(imagelist)
    X = standardization_data(X)
    X, y = shuffle_array(X, y)
    X_train, X_test, y_train, y_test = TT_split(X,y)
    X_train, X_test, y_train = data_augmentation(X_train, X_test, y_train)
    X_train, y_train = shuffle_array(X_train, y_train)
    train(X_train, y_train)
    performance = test(X_test, y_test)
    print("Metrics report on testing set: {}".format(performance))

if __name__ == '__main__':
	main()

