import math

_SCHEMA_FAMILY_TAG = 'linear_proportional_1d'
_CORE_LAW_EXPR = "(1.200918534 + 0.849939048 * f['load_n'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.200918534 + 0.849939048 * f['load_n'])
    return 0 + (1) * (1) * latent
