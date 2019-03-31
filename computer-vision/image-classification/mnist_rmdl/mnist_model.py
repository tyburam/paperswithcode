import tensorflow as tf
from sklearn.metrics import accuracy_score
from .dnn import DNN
from .cnn import CNN
from .rnn import RNN


# Implementing paper: RMDL: Random Multimodel Deep Learning for Classification
# paper authors:  Kamran Kowsari • Mojtaba Heidarysafa • Donald E. Brown • Kiana Jafari Meimandi • Laura E. Barnes
# available at: https://arxiv.org/pdf/1805.01890v2.pdf
# original source code:  https://github.com/kk7nc/RMDL
class MnistModel:
    def __init__(self, batch_size=128, networks=[0, 0, 3], epochs=[100, 100, 100]):
        self.models = []
        self.epochs = epochs
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((60000, 28, 28, 1))
        x_test = x_test.reshape((10000, 28, 28, 1))
        self.x_train, self.x_test = x_train / 255.0, x_test / 255.0
        self.y_train, self.y_test = y_train, y_test
        self.batch_size = batch_size

        shape = (28, 28, 1)
        num_classes = 10

        # DNN
        for i in range(networks[0]):
            self.models.append(DNN(shape, num_classes))
            self.models[len(self.models) - 1].compile(optimizer='adam',
                                                      loss='sparse_categorical_crossentropy',
                                                      metrics=['accuracy'])

        # RNN
        for i in range(networks[1]):
            self.models.append(RNN(shape, num_classes))
            self.models[len(self.models) - 1].compile(optimizer='adam',
                                                      loss='sparse_categorical_crossentropy',
                                                      metrics=['accuracy'])

        # CNN
        for i in range(networks[2]):
            self.models.append(CNN(shape, num_classes))
            self.models[len(self.models) - 1].compile(optimizer='adam',
                                                      loss='sparse_categorical_crossentropy',
                                                      metrics=['accuracy'])

    def train(self):
        for i in range(len(self.models)):
            self.models[i].fit(self.x_train, self.y_train)

    def score(self):
        probs, scores = [], []
        for i in range(len(self.models)):
            probs.append(self.models[i](self.x_test))
            scores.append(accuracy_score(self.y_test, probs[len(probs) - 1]))

        final_y = 0
