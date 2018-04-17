# Face_identification

![Alt Text 1](https://raw.github.com/ethilliez/Face_identification/master/Figures/Elo_4.jpg)![Alt Text 2](https://raw.github.com/ethilliez/Face_identification/master/Figures/Elo_13.jpg) ![Alt Text 3](https://raw.github.com/ethilliez/Face_identification/master/Figures/Tash_11.jpg)![Alt Text 4](https://raw.github.com/ethilliez/Face_identification/master/Figures/Tash_14.jpg)

## Description:
Algorithm performing face identification via supervised learning using a Convolutional Neural Network (CNN). The CNN was trained using a set of images featuring two different persons, which the CNN must learn to distinguish. Examples of training data are shown above.

## Personal development goals:
- Practising developing a Convolutional Neural Network using [Keras](https://github.com/fchollet/keras).
- Playing with the [Image Augmentation tool](https://github.com/ethilliez/Image_augmentation) I previously developed.
- Preparing a model to recognize people for a future real-time person identification tool.

## Status of development:
- :white_check_mark: Code Skeletion created 
- :white_check_mark: Pre-processing implemented (including face cropping, data augmentation standardization)
- :white_check_mark: Training implemented
- :white_check_mark: Testing implemented
- :white_check_mark: Raw performance in testing (no fine-tuning)

## Requirements:
The main librairies required are: `numpy`, `scipy`, `sklearn`, `keras` and `itertools`. They can be installed using `pip install` or `conda install`.

## Execution:
1. Firsly, in `define_parameters.py`:
- update the path to the folder containing the images and future model: `DATA_PATH` and `MODEL_PATH`
- enter the CNN parameters in `CNN_parameters`
- select if plotting confusion matrix in `testing_options`

2. Executing the main script: `python3 main.py` 

## Raw performance:
- 140 original images (70 per class) augmented to 1800
- train/test dataset split 80%/20%
- training on 75 epochs, best validation loss =  and validation accuracy =  (see `define_paramters.py` for all hyperparameters)
- testing on images, testing accuracy 
- see confusion matrix obtained on the testing set

![Alt Text 1](https://raw.github.com/ethilliez/Face_identification/master/Figures/Confusion_matrix.jpg)!

