import numpy as np

def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-x*a))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x, a=1):
    return np.tanh(x*a)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1.0, 0.0)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def d_softplus(x):
    return sigmoid(x)


activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "softplus": softplus
}