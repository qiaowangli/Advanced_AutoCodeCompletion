#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense, RNN, Dropout
from sklearn.model_selection import train_test_split


"""
Last edited by   : Shawn
Last edited time : 27/11/2021
Version Status: dev
TO DO: Create RNN
This is based on the work that Braiden has done
"""


def rnn(x, y):
    """
    @ input : x - inputs, and y - labels
    @ output: RNN models
    To get x and y
    x, y = data_prep.get_input_vectors_and_labels('NN_input.txt', 'vector.csv')
    """

    # Separate data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # Further separate the training data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    # Model 1 - Inspired by: https://www.youtube.com/watch?v=iMIWee_PXl8
    RNN1 = Sequential()
    # Each feature vector is a vector with 15 elements where each element is a vector of length 34
    RNN1.add(RNN(1, batch_input_shape=(None, 15, 34), return_sequences=True))
    RNN1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    history1 = RNN1.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    # Model 2 - Inspired by: https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
    RNN2 = Sequential()
    RNN2.add(RNN(32, return_sequences=True,
                 batch_input_shape=(None, 15, 34)))  # returns a sequence of vectors of dimension 32
    RNN2.add(RNN(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    RNN2.add(RNN(32))  # return a single vector of dimension 32
    RNN2.add(Dense(10, activation='softmax'))
    RNN2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    history2 = RNN2.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    return history1, history2
