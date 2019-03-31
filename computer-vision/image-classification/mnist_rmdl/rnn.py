import tensorflow as tf
import random
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense


class RNN(tf.keras.Model):
    def __init__(self, shape, number_of_classes, min_nodes_rnn=1, max_nodes_rnn=5, dropout=0.05):
        super(RNN, self).__init__()

        values = list(range(min_nodes_rnn, max_nodes_rnn))
        nodes = random.choice(values)

        self.input = Input(shape=shape)
        self.td = TimeDistributed(LSTM(nodes, recurrent_dropout=dropout))
        self.l1 = LSTM(nodes, recurrent_dropout=dropout)
        self.d = Dense(number_of_classes, activation='softmax')

    def call(self, x):
        x = self.input(x)
        x = self.td(x)
        x = self.l1(x)

        return self.d(x)
