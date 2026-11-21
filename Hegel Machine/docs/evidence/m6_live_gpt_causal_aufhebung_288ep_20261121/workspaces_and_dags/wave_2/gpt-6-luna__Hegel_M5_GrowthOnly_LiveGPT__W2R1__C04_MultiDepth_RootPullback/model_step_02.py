import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    E = np.array([r['u_indoor'] + r['p_weather'] * r['u_outdoor'] for r in rows], dtype=float)
    impact = np.array([r['impact_idx'] for r in rows], dtype=float)
    area = np.array([r['a_foot'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        gain, half_sat, loss0, loss_area = p
        pred = gain * E / (half_sat + E) - impact * (loss0 + loss_area * area)
        return pred - y

    p0 = [8.0, 1.2, 0.0, 3.5]
    result = least_squares(residual, p0, bounds=([0.0, 1e-6, -20.0, -20.0], [30.0, 30.0, 20.0, 20.0]))
    return result.x

def predict(row):
    p = fit(_MODEL_TRAIN_ROWS)
    gain, half_sat, loss0, loss_area = p
    E = row['u_indoor'] + row['p_weather'] * row['u_outdoor']
    return float(gain * E / (half_sat + E) - row['impact_idx'] * (loss0 + loss_area * row['a_foot']))