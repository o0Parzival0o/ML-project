import numpy as np

def MSE(o, t):
    l = t.shape[0]
    return 1/l * np.sum(np.sum((o-t)**2, axis=1), axis=0)

def MEE(o, t):
    l = t.shape[0]
    return 1/l * np.sum(np.sqrt(np.sum((o-t)**2, axis=1)), axis=0)


losses_functions ={
    "MSE": MSE,
    "MEE": MEE
}