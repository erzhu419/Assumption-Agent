import math

_SCHEMA_FAMILY_TAG = 'rational_1d'
_CORE_LAW_EXPR = "(f['u'] / (0.1858773254 * f['u'] + 1e-05))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['u'] / (0.1858773254 * f['u'] + 1e-05))
    return 0 + (1) * (1) * latent
