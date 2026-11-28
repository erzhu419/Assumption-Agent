import numpy as np
from scipy.optimize import minimize_scalar


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    u, v = panel['u'], panel['v2']
    v_mean = v.mean(axis=0)
    centered_v = v - v_mean

    def fit(frequency):
        features = np.cos(frequency * u)
        mean = features.mean(axis=0)
        centered = features - mean
        left, singular, right = np.linalg.svd(centered.T @ centered_v)
        mixing = (left @ right) * (singular.sum() / np.sum(centered**2))
        error = np.sum((centered @ mixing - centered_v)**2)
        return error, mixing, v_mean - mean @ mixing

    # Profile out the shared-scale orthogonal map before fitting the frequency.
    grid = np.linspace(1.0, 4.0, 31)
    best = int(np.argmin([fit(frequency)[0] for frequency in grid]))
    lower = grid[max(0, best - 1)]
    upper = grid[min(len(grid) - 1, best + 1)]
    result = minimize_scalar(lambda frequency: fit(frequency)[0],
                             bounds=(lower, upper), method='bounded',
                             options={'xatol': 1e-9})
    frequency = min((grid[best], result.x), key=lambda value: fit(value)[0])
    _, mixing, bias = fit(frequency)
    return {'b': np.cos(frequency * panel['z']) @ mixing + bias}
