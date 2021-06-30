import numpy as np

# normalization
def normalize_input(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        print(x.shape)
        raise ValueError('Dimention should be 3 or 4')
    x = x.astype('float32')
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    return (x - mean) / std_adj