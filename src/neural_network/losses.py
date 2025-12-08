import numpy as np

def MSE(o, t):
    t = t.astype(float)
    diff = o - t
    return np.mean(diff**2)

def MEE(o, t):
    t = t.astype(float)
    diff = o - t
    return np.mean(np.sqrt(np.sum(diff**2, axis=1)))

def binary_crossentropy(o, t):
    t = t.astype(float)
    eps = 1e-8
    for k in o:
        k = k + (-eps if k == 1. else eps if k == 0. else 0.)
    bce = - (t * np.log(o) + (1 - t) * np.log(1 - o))
    return np.mean(bce)

losses_functions = {
    "MSE": MSE,
    "MEE": MEE,
    "binary_crossentropy": binary_crossentropy
}