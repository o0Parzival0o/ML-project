import numpy as np


def linear(x, a=1.):
    return a*x

def d_linear(x, a=1.):
    return a

def sigmoid(x, a=1.):
    return 1. / (1. + np.exp(-x*a))

def d_sigmoid(x, a=1.):
    s = sigmoid(x, a)
    return a * s * (1. - s)

def tanh(x, a=1.):
    return np.tanh(x*a)

def d_tanh(x, a=1.):
    return a * (1. - np.tanh(x*a)**2.)

def relu(x, a=0.):
    return np.maximum(0., x) + (np.minimum(0., a*x) if a != 0. else 0.)

def d_relu(x, a=0.):
    return np.where(x > 0., 1., a)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.)

def d_softplus(x):
    return sigmoid(x)


activation_functions = {
    "linear": [linear, d_linear],
    "sigmoid": [sigmoid, d_sigmoid],
    "tanh": [tanh, d_tanh],
    "relu": [relu, d_relu],
    "softplus": [softplus, d_softplus]
}