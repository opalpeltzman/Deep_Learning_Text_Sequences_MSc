import glob
import os

import loglinear as ll
import random
import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return np.array(features)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pass
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def set_path(name: str) -> str:
    """
    this function creates absolute path for a given file or folder
    :param name - file or folder name
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'{name}'))
    return path

def read_data(folder: str, fname: str) -> list:
    """
        this function extracts file data
        :param folder - folder name in order to set the path
        :param fname - file name to read
    """
    data = []
    path = set_path(folder)
    file_name = os.path.join(path, fname)
    with open(file_name, encoding='utf8') as f:
        file_data = f.readlines()

    for line in file_data:
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    learning_rate = 0.01
    num_iterations = 1000
    dev_data = read_data(folder='data', fname='dev')
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

