# Face_identification

## Description:
Algorithm performing face identification via supervised learning using a Convolutional Neural Network (CNN). The CNN was trained using a set of images featuring two different persons, which the CNN must learn to distinguish. 

## Personal development goals:
- Practising developing a Convolutional Neural Network using [Keras](https://github.com/fchollet/keras).
- Playing with the [Image Augmentation tool](https://github.com/ethilliez/Image_augmentation) I previously developed.
- Preparing a model to recognize people for a future real-time person identification tool.
- Practising performing the same task with Tensorflow/Caffe ?

## Status of development:
For Keras implementation:
- Code Skeletion created
- Pre-processing implemented
- Training implemented
- Testing implemented
- Raw performance in testing (no fine-tuning)

## Requirements:
The main librairies required are: `numpy`, `scipy`, `sklearn`, `keras` and `itertools`. They can be installed using `pip install` or `conda install`.

## Execution:
1. Firsly, in `define_parameters.py`:
- update the path to the folder containing the images and future model: `DATA_PATH` and `MODEL_PATH`
- enter the CNN parameters in `CNN_parameters`
- select if plotting confusion matrix in `testing_options`

2. Executing the main script: `python3 main.py` 

## Raw performance:
- 100 original images (50 per class) augmented to 1600
- train/test dataset split 80%/20%

