import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import collections
import string

def window_transform_series(series, window_size):
    """
    Transforms the input series 
    and window-size into a set of input/output pairs for use with our RNN model
    """
    # containers for input/output pairs
    X = []
    y = []
    for idx in range(0, len(series) - window_size):
        X.append(series[idx:idx+window_size])
        y.append(series[idx+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def build_part1_RNN(window_size):
    """
    Build an RNN to perform regression on our time series input/output data using Keras
    with a fixed size of 5 nodes in the LSTM and one single output node with no activation function
    so that it can output a value matching the time series.
    """
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size,1)))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    """
    Return the text input with only ascii lowercase and the punctuation given below included.
    This will also exclude numbers, special foreign characters, ), $, % etc.
    """
    punctuation = ['!', ',', '.', ':', ';', '?']
    freq_dict = collections.Counter(text)
    for character in freq_dict:
        if (character not in punctuation) and (character not in string.ascii_lowercase):
            print("Removing character: ", character)
            text = text.replace(character, ' ')
    return text

def window_transform_text(text, window_size, step_size):
    """
    Transform the input text and window-size into a set of input/output pairs for use with our RNN model
    We move the window by the parameter step size so that we don't have to generate too many pairs.
    """
    # containers for input/output pairs
    inputs = []
    outputs = []

    for idx in range(0, len(text) - window_size, step_size):
        inputs.append(text[idx:idx+window_size])
        outputs.append(text[idx+window_size])

    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    """
    Build the required RNN model: 
    a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
    """
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model

