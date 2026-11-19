import math

_SCHEMA_FAMILY_TAG = 'rational_monod_1d'
_CORE_LAW_EXPR = "(f['tmp_bar'] / (0.2649942299 * f['tmp_bar'] + 1.695324095))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['tmp_bar'] / (0.2649942299 * f['tmp_bar'] + 1.695324095))
    return 0 + (1) * (1) * latent
