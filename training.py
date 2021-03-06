from define_parameters import CNN_parameters, paths
from keras import backend, regularizers, optimizers
from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class train_CNN:
    def __init__(self):
        self.epochs = CNN_parameters.EPOCHS
        self.nb_batch = CNN_parameters.BATCH_SIZE
        self.dropout_coef = CNN_parameters.DROPOUT
        self.acti_neuron = CNN_parameters.ACTIVATION
        self.opti = CNN_parameters.OPTIMIZER
        self.lnr = CNN_parameters.LEARNING_RATE
        self.l2regul = CNN_parameters.L2_REGUL
        self.nb_class = CNN_parameters.NB_CLASS
        self.model_output_path = paths.MODEL_PATH
        self.model_name = paths.MODEL_NAME

    def CNN_model(self, shape_input):
        seed(1)
        set_random_seed(2)
    # Build Model
        model = Sequential()
        model.add(Conv2D(32,kernel_size=(10,10), strides=(2, 2), activation = self.acti_neuron, input_shape = (shape_input), kernel_regularizer = regularizers.l2(self.l2regul)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(self.dropout_coef))

        #model.add(Conv2D(32,(4,4), activation = self.acti_neuron, kernel_regularizer = regularizers.l2(self.l2regul)))
        #model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3), strides=(1, 1), activation = self.acti_neuron, kernel_regularizer = regularizers.l2(self.l2regul)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5*self.dropout_coef))

        model.add(Flatten())
        model.add(Dense(128, activation = self.acti_neuron))
        model.add(Dropout(int(1.0*self.dropout_coef)))

        if(self.opti == 'adam'): 
            optimizer = optimizers.Adam(lr = self.lnr)

        if(self.nb_class == 2):
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['binary_accuracy']) 
        elif(self.nb_class > 2):
            model.add(Dense(self.nb_class, activation='softmax'))
            model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['categorical_accuracy']) 

        return model

    def main_train(self, X_train, y_train):
    # Perform Training and save final model
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)  
        shape_input = X_train[0].shape
        model = self.CNN_model(shape_input)
        min_loss = 1.0
        max_accu = 0.5
        count = 0

        for i in range(self.epochs):
            history = model.fit(X_train, y_train, batch_size = self.nb_batch, shuffle= 'batch', epochs=1, validation_split = 0.05)
            if(history.history['val_loss'][0] <= min_loss and history.history['val_binary_accuracy'][0] >= 0.9*max_accu):
                min_loss = history.history['val_loss'][0]
                max_accu = history.history['val_binary_accuracy'][0]
                logger.info((" New saved weights with loss: ", np.round(min_loss,3), "and accuracy: ", np.round(max_accu,3)))
                model.save(self.model_output_path+self.model_name, overwrite = True)
                count = 0
            else: 
                count = count + 1
                if(count == 10):
                    logger.info("    Not learning anymore. Stopping training.")
                    break

