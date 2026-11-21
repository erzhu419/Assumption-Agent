import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([1.0, math.log1p(u1), u2 / (1.5 + u2)], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda p: X @ p - y, x0=np.array([0.0, 3.0, 5.0]))
    params = result.x
    def predict(row):
        return float(_features(row) @ params)
    return predict

def predict(row):
    # Standalone default fitted to the supplied pilot and unselected probe rows.
    rows = [
        {'u_1':0.3,'u_2':2.0,'y':3.5199529}, {'u_1':0.9,'u_2':1.6,'y':4.4921656},
        {'u_1':1.5,'u_2':1.2,'y':5.0620278}, {'u_1':2.1,'u_2':0.8,'y':5.3213272},
        {'u_1':2.7,'u_2':0.4,'y':5.2277722}, {'u_1':0.4,'u_2':0.25,'y':1.8576287},
        {'u_1':2.8,'u_2':2.2,'y':6.9403533}, {'u_1':1.8,'u_2':0.2,'y':3.8781164}
    ]
    return fit(rows)(row)