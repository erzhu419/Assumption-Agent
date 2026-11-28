import numpy as np


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def run_node(inputs: dict) -> dict:
    first = inputs['node_02']
    coef = np.linalg.lstsq(_affine(first['b_u']), first['b_v'], rcond=None)[0]
    return {'z_tr': _affine(first['u_tr']) @ coef,
            'z_te': _affine(first['u_te']) @ coef}
