"""Build a common bilinear feature basis for both frozen readouts."""
import numpy as np


def _design(x, c):
    cross = (x[:, :, None] * c[:, None, :]).reshape(len(x), x.shape[1] * c.shape[1])
    return np.column_stack((np.ones(len(x)), x, c, cross))


def run_node(inputs: dict) -> dict:
    coords = inputs['node_00']
    return dict(d_train=_design(coords['x_train'], coords['c_train']),
                d_test=_design(coords['x_test'], coords['c_test']))
