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
from keras.layers import Dense, LSTM, Dropout, Masking
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

    # Create an instance of the Sequential class
    model = Sequential()
    model.add(Masking(mask_value= [0]*34, input_shape=x_train.shape[1:]))
    model.add(LSTM(2500, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    model.add(Dropout(0.2))
    model.add(LSTM(2500, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    model.add(Dropout(0.2))
    model.add(LSTM(2500, activation="relu", input_shape=x_train.shape[1:], return_sequences=True ))
    model.add(Dropout(0.2))
    model.add(Dense(7374, activation="softmax", input_shape=x_train.shape[1:]))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val))
    prediction = history.predict(x_test)


    return (model, history, prediction)




