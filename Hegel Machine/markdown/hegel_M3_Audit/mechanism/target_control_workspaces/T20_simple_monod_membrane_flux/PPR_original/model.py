import math

_SCHEMA_FAMILY_TAG = 'congestion_rollover_1d'
_CORE_LAW_EXPR = "(f['tmp_bar'] / (0.2654566637 * f['tmp_bar'] + 1.693464534 + 1e-06 * (f['tmp_bar'] ** 2)))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['tmp_bar'] / (0.2654566637 * f['tmp_bar'] + 1.693464534 + 1e-06 * (f['tmp_bar'] ** 2)))
    return 0 + (1) * (1) * latent
