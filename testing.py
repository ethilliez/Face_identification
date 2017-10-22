from define_parameters import CNN_parameters, paths, testing_options
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import logging

class test_CNN:

    def __init__(self):
        self.model_output_path = paths.MODEL_PATH
        self.model_name = paths.MODEL_NAME
        self.nb_batch = CNN_parameters.BATCH_SIZE
        self.nb_class = CNN_parameters.NB_CLASS
        self.plot = testing_options.PLOT
        self.class_names = testing_options.CLASS_NAMES

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def main_test(self, X_test, y_test):
        # Load CNN model
        model = load_model(self.model_output_path+self.model_name)
        # Perform testing
        y_predict = model.predict(X_test, batch_size = self.nb_batch)
        # Compute metrics
        metrics_report = classification_report(y_test, np.round(y_predict))
        # Plot confusion matrix if required
        if(self.plot):
            cnf_matrix = confusion_matrix(y_test, np.round(y_predict))
            fig = plt.figure()
            self.plot_confusion_matrix(cnf_matrix, classes = self.class_names, normalize=True,
                      title='Normalized confusion matrix')
            fig.savefig("Confusion_matrix.png",format='png', dpi=300)
            fig.clf()
        return metrics_report




