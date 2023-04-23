import numpy as np
import loglinear as ll
import math

STUDENT={'name': 'Opal Peltzman',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    h_l = x
    for inx in range(0, len(params), 2):
        W, b = params[inx], params[inx+1]
        h_l = np.tanh(np.dot(h_l, W) + b)

    probs = ll.classifier_output(h_l, params[-2:])
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    U, b_tag = params[-2:]
    y_hat = classifier_output(x=x, params=params)
    y_vec = np.zeros(U.shape[1])
    y_vec[y] = 1

    loss = -math.log(y_hat[y])
    zn = [x]
    an = [x]
    for inx in range(0, len(params), 2):
        W, b = params[inx], params[inx + 1]
        zn.append(zn[-1].dot(W) + b)
        an.append(np.tanh(zn[-1]))  # output layer n

    gb_tag = y_hat - y_vec
    gU = np.array([an[-2]]).transpose().dot(np.array([gb_tag]))
    grads = [gU, gb_tag]

    for inx in range(0, len(params), 2)[::-1]:
        gb = grads[1].dot(params[inx + 1].transpose()) * (1 - np.power(an[inx + 1], 2))
        gW = np.array([zn[inx]]).transpose().dot(np.array([gb]))
        grads.insert(0, gb)
        grads.insert(0, gW)
    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        params.append(np.random.randn(dims[i], dims[i+1]) / np.sqrt(dims[i]))
        params.append(np.zeros(dims[i+1]))
    return params

