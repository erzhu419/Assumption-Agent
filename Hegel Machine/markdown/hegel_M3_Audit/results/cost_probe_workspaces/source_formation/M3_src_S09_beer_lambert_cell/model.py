import math

_SCHEMA_FAMILY_TAG = 'exp_washout_decay_1d'
_CORE_LAW_EXPR = "(0.4501560654 + 4.18295927 * math.exp(-f['path_mm'] / 4.218))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.4501560654 + 4.18295927 * math.exp(-f['path_mm'] / 4.218))
    return 0 + (1) * (1) * latent
