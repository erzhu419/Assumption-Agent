import math

_SCHEMA_FAMILY_TAG = 'overdamped_charging_2d'
_CORE_LAW_EXPR = "(1.479794657 * f['bias_v'] * (1.0 - math.exp(-f['gate_s'] / 1.7)))"
_LENS = {'sign': -1.0, 'scale': 1.3, 'offset': 2.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.479794657 * f['bias_v'] * (1.0 - math.exp(-f['gate_s'] / 1.7)))
    return 2.5 + (-1) * (1.3) * latent
