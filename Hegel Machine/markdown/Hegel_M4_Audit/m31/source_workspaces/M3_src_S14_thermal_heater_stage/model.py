import math

_SCHEMA_FAMILY_TAG = 'overdamped_charging_2d'
_CORE_LAW_EXPR = "(1.649913598 * f['power_w'] * (1.0 - math.exp(-f['elapsed_s'] / 1.5)))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.649913598 * f['power_w'] * (1.0 - math.exp(-f['elapsed_s'] / 1.5)))
    return 0 + (1) * (1) * latent
