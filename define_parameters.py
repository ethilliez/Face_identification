class paths:
    DATA_PATH = '../../Data/Augmented/'  # Folder path where raw images are stored
    MODEL_PATH = 'Model/'  # Folder which contains the output images
    MODEL_NAME = 'Best_CNN_model.h5'

class CNN_parameters:
    EPOCHS = 50
    BATCH_SIZE = 32
    DROPOUT = 0.2
    ACTIVATION = 'relu'
    OPTIMIZER = 'adam'
    NB_CLASS = 2

class testing_options:
	PLOT = True
	CLASS_NAMES = ["Elo", "Tash"]
