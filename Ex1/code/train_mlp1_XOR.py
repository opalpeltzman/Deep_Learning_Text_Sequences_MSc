import os
import mlp1 as mln
import random
import numpy as np
from math import log

STUDENT = {'name': 'Opal Peltzman',
           'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return np.array(features)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if mln.predict(x=features, params=params) == label:
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
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.
            loss, grads = mln.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= grads[0] * learning_rate
            params[1] -= grads[1] * learning_rate
            params[2] -= grads[2] * learning_rate
            params[3] -= grads[3] * learning_rate

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def extract_info_from_data(data: list, lan_classes: dict, bigrams_features: dict) -> list:
    """
      this function creates a list that indicates the features for each language that
      accrues in the given data list
    """
    info = []
    for [language, biagram] in data:
        lan_features = np.zeros(len(bigrams_features))
        if biagram in bigrams_features:
            lan_features[bigrams_features[biagram]] = 1
        language_dict = lan_classes[language] if language in lan_classes else -1
        info.append([language_dict, lan_features])
    return info


def predicting_test_data(lan_classes: dict, trained_params, output_file_name: str, test_data: list):
    inv_lan_classes = {v: k for k, v in lan_classes.items()}
    pred = [inv_lan_classes[mln.predict(x=data, params=trained_params)] for _, data in test_data]

    with open(output_file_name, 'w') as f:
        f.write('\n'.join(pred))


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    learning_rate = 0.001
    num_iterations = 349

    with open('xor_data.py', encoding='utf8') as f:
        file_data = f.readlines()

    data = []
    for line in file_data:
        if line != '\n':
            label, text = line.replace('(','').strip().split(",", 1)
            label = label.replace('data =','').replace("'",'').replace(" ",'').replace("[",'').replace("]",'')
            text = text.split(")", 1)[0]
            data.append((label, text))

    _lan_classes = {'0': 0, '1': 1}
    _bigrams_features = {'[0,0]': 0, '[0,1]': 1, '[1,0]': 2, '[1,1]': 3}
    xor_data = extract_info_from_data(data, _lan_classes, _bigrams_features)
    print()
    in_dim = len(_bigrams_features)
    hid_dim = int(log(len(_bigrams_features)))
    out_dim = len(_lan_classes)

    params = mln.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(xor_data, xor_data, num_iterations, learning_rate, params)


