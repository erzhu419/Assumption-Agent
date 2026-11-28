import numpy as np
from scipy.linalg import solve


def run_node(inputs: dict) -> dict:
    features = inputs['node_00']
    trajectory = inputs['node_02']['state_pred']
    scores = features['scores']
    query_scores = trajectory @ features['modes'].T
    sy, qy = features['sy'], features['qy']

    # Use matched training/query features for both calibrated corrections.
    gram = features['K'] + 0.03 * (sy @ sy.T) + 0.01 * (scores @ scores.T)
    cross = features['Kq'] + 0.03 * (qy @ sy.T) + 0.01 * (query_scores @ scores.T)
    target = features['y_centered'] + features['y_mean']
    alpha = solve(gram + 0.004 * np.eye(len(gram)), target, assume_a='pos')
    return {'y_hat': cross @ alpha}
