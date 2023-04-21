import numpy as np
import loglinear as ll
import math

STUDENT={'name': 'Opal Peltzman',
         'ID': 'YOUR ID NUMBER'}


def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b = params[0], params[1]
    h_l = np.tanh(np.dot(x, W) + b)
    probs = ll.classifier_output(h_l, [params[2], params[3]])
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params

    y_hat = classifier_output(x=x, params=params)
    # y_vec equals 1 in the correct label position (y)
    y_vec = np.zeros(U.shape[1])
    y_vec[y] = 1

    loss = -math.log(y_hat[y])
    z1 = np.tanh(x.dot(W) + b)   # output layer 1 (hidden layer)
    gb_tag = y_hat - y_vec
    gU = np.array([z1]).transpose().dot(np.array([gb_tag]))
    gb = gb_tag.dot(U.transpose()) * (1 - np.power(z1, 2))
    gW = np.array([x]).transpose().dot(np.array([gb]))

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """


    params = [W, b, U, b_tag]
    return params

