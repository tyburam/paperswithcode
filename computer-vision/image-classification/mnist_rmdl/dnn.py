import tensorflow as tf
import random
from tensorflow.keras.layers import Flatten, Dense, Dropout


class DNN(tf.keras.Model):
    def __init__(self, shape, number_of_classes, min_hidden_layer_dnn=1, max_hidden_layer_dnn=8,
                 min_nodes_dnn=128, max_nodes_dnn=1024, dropout=0.05):
        super(DNN, self).__init__()

        values = list(range(min_nodes_dnn, max_nodes_dnn))
        number_of_nodes = random.choice(values)
        l_values = list(range(min_hidden_layer_dnn, max_hidden_layer_dnn))
        n_layers = random.choice(l_values)

        self.flat = Flatten(input_shape=shape)
        self.d0 = Dense(number_of_nodes, activation='relu')
        self.drop = Dropout(dropout)

        self.n_dense = []
        for i in range(n_layers - 1):
            number_of_nodes = random.choice(values)
            self.n_dense.append(Dense(number_of_nodes, activation='relu'))
            self.n_dense.append(Dropout(dropout))

        self.dN = Dense(number_of_classes, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.d0(x)
        x = self.drop(x)

        for i in range(len(self.n_dense)):
            x = self.n_dense[i](x)

        return self.dN(x)
