import os.path as osp

import numpy as np
import torch_geometric.transforms as T
from scipy.optimize import linear_sum_assignment
from torch_geometric.datasets import Planetoid


def map_labels(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind


def get_planetoid_dataset(name):
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name, T.NormalizeFeatures())

    return dataset
