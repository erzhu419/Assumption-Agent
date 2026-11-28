import numpy as np


def run_node(inputs: dict) -> dict:
    spatial=inputs['node_00']; temporal=inputs['node_01']
    y=spatial['regression']
    weights=np.linalg.lstsq(spatial['train_features'],temporal['scores'],rcond=1e-10)[0]
    state=(spatial['test_features'] @ weights) @ temporal['basis'].T
    final=state[:,-1]
    return {'y_hat':y.copy(),'state_pred':state,
            'diagnostics_vec':np.array([y.mean(),y.std(),final.mean(),final.std()])}
