import numpy as np


def softmax(z):
    assert len(z.shape) == 2


    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] 
    return e_x / div


x1 = np.array([[xxx, xxx, xxx, xxx]])
softmax(x1)
