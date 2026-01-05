import numpy as np


def MSE(o, t):
    t = t.astype(float)

    eps = 1e-15
    o_clip = np.clip(o, eps, 1-eps)
    
    diff = o_clip - t
    return np.mean(diff**2)

def MEE(o, t):
    t = t.astype(float)
    
    eps = 1e-15
    o_clip = np.clip(o, eps, 1-eps)
    
    diff = o_clip - t
    return np.mean(np.sqrt(np.sum(diff**2, axis=1)))

def binary_crossentropy(o, t):
    t = t.astype(float)
    eps = 1e-15
    o_clip = np.clip(o,eps,1-eps)

    bce = - (t * np.log(o_clip) + (1 - t) * np.log(1 - o_clip))
    return np.mean(bce)

losses_functions = {
    "MSE": MSE,
    "MEE": MEE,
    "binary_crossentropy": binary_crossentropy
}
