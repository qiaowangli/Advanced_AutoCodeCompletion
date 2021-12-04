#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Masking, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

    sequence_length = len(x_train[0])
    sequence_element_length = len(x_train[0][0])

    # Create an instance of the Sequential class
    model = Sequential()

    model.add(Masking(mask_value=[0] * 34, input_shape=(sequence_length, sequence_element_length)))
    model.add(SimpleRNN(10000, activation="relu", input_shape=(sequence_length, sequence_element_length),
                        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(10000, activation="relu", input_shape=(sequence_length, sequence_element_length),
                        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(10000, activation="relu", input_shape=(sequence_length, sequence_element_length),
                        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(7374, activation="softmax", input_shape=(sequence_length, sequence_element_length)))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val))
    prediction = history.predict(x_test)

    return model, history, prediction


def main():
    inputs = []
    labels = []

    # Get the word embedding table as a df
    word_embedding_df = pd.read_csv("pca_lookup_table.csv", header=None)

    file = open("NN_input.txt")
    sequence_list = []
    for sequence in file:
        sequence = [int(x) for x in sequence.strip().strip('][').split(',')]
        sequence_list.append(sequence)
    file.close()

    for seq in sequence_list:
        # Replace the current integer with its corresponding vector in the word embedding table if > 0,
        # else use vector of all 0's
        inputs.append([list(word_embedding_df.loc[val - 1]) if val > 0 else [0] * 34 for val in seq[:-1]])
        # Store the last integer in each sequence as the label
        # labels.append([list(word_embedding_df.loc[val - 1]) if val > 0 else [0] * 34 for val in seq[-1:]])
        # one-hot
        labels.append([[1 if seq[-1] - 1 == i else 0 for i in range(7374)]])

    # Convert the inputs and labels to numpy arrays
    inputs = np.array(inputs, dtype=float)
    labels = np.array(labels, dtype=float)

    model, history, predict = rnn(inputs, labels)


    return


main()
