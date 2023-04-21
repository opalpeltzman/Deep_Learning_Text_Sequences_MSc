import glob
import os

import loglinear as ll
import random
import numpy as np

STUDENT={'name': 'Opal Peltzman',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return np.array(features)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if ll.predict(x=features, params=params) == label:
            good += 1
        else:
            bad += 1
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
            params[0] -= grads[0] * learning_rate
            params[1] -= grads[1] * learning_rate

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


def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]


def create_train_data(train_data: list) -> tuple:
    """
        this function creates language clusters dictionary and
        biagram features dictionary
        :param train_data - train data from train file provided
    """
    lan_classes = {lan: inx for inx, lan in enumerate(set(map(lambda x: x[0], train_data)))}
    bigram_lists = [bigram for _, bigram in train_data]
    flat_bigram_set = set([item for sublist in bigram_lists for item in sublist])
    bigrams_features = {bigram: inx for inx, bigram in enumerate(flat_bigram_set)}

    return lan_classes, bigrams_features


def extract_info_from_data(data: list, lan_classes: dict, bigrams_features: dict) -> list:
    """
      this function creates a list that indicates the features for each language that
      accrues in the given data list
    """
    info = []
    for [language, biagram] in data:
        lan_features = np.zeros(len(bigrams_features))
        for b in biagram:
            if b in bigrams_features:
                lan_features[bigrams_features[b]] = 1
        language_dict = lan_classes[language] if language in lan_classes else -1
        info.append([language_dict, lan_features])
    return info


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    learning_rate = 0.01
    num_iterations = 10

    TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data(folder='data', fname='train')]
    DEV = [(l, text_to_bigrams(t)) for l, t in read_data(folder='data', fname='dev')]
    _lan_classes, _bigrams_features = create_train_data(train_data=TRAIN)

    train_data = extract_info_from_data(TRAIN, _lan_classes, _bigrams_features)
    dev_data = extract_info_from_data(DEV, _lan_classes, _bigrams_features)
    in_dim = len(_bigrams_features)
    out_dim = len(_lan_classes)

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

