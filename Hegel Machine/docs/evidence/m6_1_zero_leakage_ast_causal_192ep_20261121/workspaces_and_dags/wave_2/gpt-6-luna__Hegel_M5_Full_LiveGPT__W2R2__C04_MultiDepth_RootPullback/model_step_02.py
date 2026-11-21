import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    E = np.array([float(r['u_outdoor']) * float(r['p_weather']) + float(r['u_indoor']) for r in rows])
    impact = np.array([float(r['impact_idx']) * float(r['a_foot']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        g, k, c = p
        return g * E / (k + E) - c * impact - y
    result = least_squares(residual, x0=[8.0, 1.0, 5.0], bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    return tuple(float(v) for v in result.x)

def predict(row):
    g, k, c = fit([{'u_outdoor':2.5,'u_indoor':0.4,'p_weather':0.12,'a_foot':1.15,'impact_idx':0.25,'y':3.0157013}, {'u_outdoor':0.5,'u_indoor':1.6,'p_weather':0.12,'a_foot':1.1,'impact_idx':0.3,'y':4.6324856}, {'u_outdoor':3.2,'u_indoor':1.0,'p_weather':0.12,'a_foot':1.85,'impact_idx':0.9,'y':2.0487543}, {'u_outdoor':1.0,'u_indoor':2.6,'p_weather':0.12,'a_foot':1.2,'impact_idx':0.2,'y':5.4636459}, {'u_outdoor':1.8,'u_indoor':2.2,'p_weather':0.12,'a_foot':1.95,'impact_idx':0.85,'y':0.0389176}, {'u_outdoor':0.2,'u_indoor':3.0,'p_weather':0.12,'a_foot':1.05,'impact_idx':0.15,'y':5.8380475}, {'u_outdoor':0.4,'u_indoor':2.8,'p_weather':0.12,'a_foot':1.48,'impact_idx':0.95,'y':0.8880277}, {'u_outdoor':2.0,'u_indoor':0.0,'p_weather':0.12,'a_foot':1.2,'impact_idx':0.0,'y':1.4311427}])
    E = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    return float(g * E / (k + E) - c * float(row['impact_idx']) * float(row['a_foot']))