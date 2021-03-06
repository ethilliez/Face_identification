class paths:
    DATA_PATH = '../../Data/EloTash/'  # Folder path where raw images are stored
    MODEL_PATH = 'Model/'  # Folder which contains the output images
    MODEL_NAME = 'Best_CNN_model.h5' # Name of saved model in MODEL_PATH

class image_parameters:
    SIZE_IMAGE = 120  # Size of output images in pixels
    TRANSLATION = 0.05  # Fraction of the image size to be translated
    ROTATION = 12.0  # Angle of rotation in degrees
    SHEARING = 0.07  # Shearing factor
    CONTRAST = [0.33,15]  # Gain and Bias for contrast adjustement 

class CNN_parameters:
    EPOCHS = 80  # Nb of epochs for CNN training
    BATCH_SIZE = 40 # Nb of images per batch of training
    DROPOUT = 0.40  # Dropout factor
    ACTIVATION = 'relu' # Activation layer used
    OPTIMIZER = 'adam' # Optimizer used in CNN training
    LEARNING_RATE = 0.0001# Learning rate of CNN
    L2_REGUL = 0.015 # L2 regularization
    NB_CLASS = 2 # Binary class problem or multiclass

class testing_options:
	PLOT = True # Plot the confusion matrix in testing
	CLASS_NAMES = ["Elo", "Tash"] # Name of the different classes
