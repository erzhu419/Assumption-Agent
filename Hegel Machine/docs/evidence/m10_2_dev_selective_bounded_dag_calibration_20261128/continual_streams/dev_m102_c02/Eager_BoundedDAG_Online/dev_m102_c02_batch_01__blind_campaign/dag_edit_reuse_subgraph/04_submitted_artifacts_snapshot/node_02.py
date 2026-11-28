import numpy as np


def run_node(inputs: dict) -> dict:
    stage = inputs['node_01']
    beta = np.asarray(stage['beta'], dtype=float)
    fitted = np.asarray(stage['fitted'], dtype=float)
    design = np.asarray(stage['new_design'], dtype=float)
    _, _, modes = np.linalg.svd(fitted, full_matrices=False)
    basis = modes[:2].T
    return {'state_pred': (design @ (beta @ basis)) @ basis.T}
