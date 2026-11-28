import numpy as np
from scipy.optimize import least_squares


def _design(u, scales):
    x = u[:, :2]
    return np.column_stack((np.ones(len(u)), u,
                            x * np.exp(-scales[:2] * x**2),
                            np.sin(scales[2:] * u[:, 2:])))


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_02']
    u, v = upstream['b_u'], upstream['b_v']

    def fit(scales):
        design = _design(u, scales)
        return design, np.linalg.lstsq(design, v, rcond=None)[0]

    def residual(scales):
        design, coef = fit(scales)
        return (design @ coef - v).ravel()

    scales = least_squares(residual, [1., 1., 3., 3.],
                           bounds=([.1, .1, 1., 1.], [5., 5., 6., 6.])).x
    _, coef = fit(scales)
    return {'z_tr': _design(upstream['u_tr'], scales) @ coef,
            'z_te': _design(upstream['u_te'], scales) @ coef}
