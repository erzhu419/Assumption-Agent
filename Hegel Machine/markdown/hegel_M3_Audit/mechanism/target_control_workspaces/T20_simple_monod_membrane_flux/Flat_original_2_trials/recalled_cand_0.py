import math

_SCHEMA_FAMILY_TAG = 'log_saturation_1d'
_CORE_LAW_EXPR = "(1e-05 + 0.9843728559 * math.log(1.0 + f['tmp_bar']))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1e-05 + 0.9843728559 * math.log(1.0 + f['tmp_bar']))
    return 0 + (1) * (1) * latent
