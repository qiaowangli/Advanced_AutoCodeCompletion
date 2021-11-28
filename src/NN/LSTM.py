#!/usr/bin/python3

"""
Last edited by   : Braiden
Last edited time : 27/11/2021
Version Status: dev
TO DO: Determine LSTM parameters
"""
# import data_prep
# from data_prep import get_input_vectors_and_labels

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

def lstm(x,y):
    """
    @ input : x - inputs, and y - labels
    @ output: LSTM models

    To get x and y
    x, y = data_prep.get_input_vectors_and_labels('NN_input.txt', 'vector.csv')
    """

    # Separate data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # Further separate the training data into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    # Model 1 - Inspired by: https://www.youtube.com/watch?v=iMIWee_PXl8
    lstm1 = Sequential()
    # Each feature vector is a vector with 15 elements where each element is a vector of length 34
    lstm1.add(LSTM((1), batch_input_shape=(None, 15, 34), return_sequences=True))
    lstm1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    history1 = lstm1.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    # Model 2 - Inspired by: https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
    lstm2 = Sequential()
    lstm2.add(LSTM(32, return_sequences=True,
                batch_input_shape=(None, 15, 34)))  # returns a sequence of vectors of dimension 32
    lstm2.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    lstm2.add(LSTM(32))  # return a single vector of dimension 32
    lstm2.add(Dense(10, activation='softmax'))
    lstm2.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    history2 = lstm2.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


    return (history1, history2)




