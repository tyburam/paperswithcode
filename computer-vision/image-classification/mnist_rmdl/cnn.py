import tensorflow as tf
import random
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.constraints import MaxNorm


class CNN(tf.keras.Model):
    def __init__(self, shape, number_of_classes, min_hidden_layer_cnn=3, max_hidden_layer_cnn=10,
                 min_nodes_cnn=128, max_nodes_cnn=512, dropout=0.05):
        super(CNN, self).__init__()

        values = list(range(min_nodes_cnn, max_nodes_cnn))
        l_values = list(range(min_hidden_layer_cnn, max_hidden_layer_cnn))
        n_layers = random.choice(l_values)
        conv_count = random.choice(values)

        self.conv0 = Conv2D(conv_count, (3, 3), padding='same', input_shape=shape, activation='relu')
        self.conv1 = Conv2D(conv_count, (3, 3), activation='relu')

        self.n_conv = []
        for i in range(n_layers):
            conv_count = random.choice(values)
            self.n_conv.append(Conv2D(conv_count, (3, 3), padding='same', activation='relu'))
            self.n_conv.append(MaxPooling2D(pool_size=(2, 2)))
            self.n_conv.append(Dropout(dropout))

        self.flat = Flatten()
        self.d0 = Dense(256, activation='relu')
        self.drop = Dropout(dropout)
        self.d1 = Dense(number_of_classes, activation='softmax', kernel_constraint=MaxNorm(3))

    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        for i in range(len(self.n_conv)):
            x = self.n_conv[i](x)

        x = self.flat(x)
        x = self.d0(x)
        x = self.drop(x)

        return self.d1(x)
