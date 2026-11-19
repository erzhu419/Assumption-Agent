import math

_SCHEMA_FAMILY_TAG = 'log_saturation_1d'
_CORE_LAW_EXPR = "(0.4009898732 + 1.949675257 * math.log(1.0 + f['baffle_cm']))"
_LENS = {'sign': -1.0, 'scale': 1.15, 'offset': 2.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.4009898732 + 1.949675257 * math.log(1.0 + f['baffle_cm']))
    return 2 + (-1) * (1.15) * latent
