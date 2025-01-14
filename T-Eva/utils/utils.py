import numpy as np
import torch

def flatten(list_):
    return [s for sublist in list_ for s in sublist]


def get_batch(X, batch_size, shuffle=False):
    data_size = len(X)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_x = X[shuffle_indices]
    else:
        shuffled_x = X

    num_batches = int(data_size / batch_size) + 1

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            tX = shuffled_x[start_index:end_index]
            tX = flatten(tX)
            yield torch.Tensor(tX)
