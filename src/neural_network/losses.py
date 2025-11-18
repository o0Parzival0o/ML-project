import numpy as np

def MSE(o, t):
    t = t.astype(float)
    diff = o - t
    return np.mean(np.sum(diff**2, axis=0))

def MEE(o, t):
    t = t.astype(float)
    diff = o - t
    return np.mean(np.sqrt(np.sum(diff**2, axis=0)))


losses_functions ={
    "MSE": MSE,
    "MEE": MEE
}