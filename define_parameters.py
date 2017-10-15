class paths:
    DATA_PATH = '../../Data/Augmented/'  # Folder path where raw images are stored
    MODEL_PATH = 'Model/'  # Folder which contains the output images
    MODEL_NAME = 'Best_CNN_model.h5' # Name of saved model in MODEL_PATH

class CNN_parameters:
    EPOCHS = 50  # Nb of epochs for CNN training
    BATCH_SIZE = 32 # Nb of images per batch of training
    DROPOUT = 0.2  # Dropout factor
    ACTIVATION = 'relu' # Activation layer used
    OPTIMIZER = 'adam' # Optimizer used in CNN training
    NB_CLASS = 2 # Binary class problem or multiclass

class testing_options:
	PLOT = True # Plot the confusion matrix in testing
	CLASS_NAMES = ["Elo", "Tash"] # Name of the different classes
