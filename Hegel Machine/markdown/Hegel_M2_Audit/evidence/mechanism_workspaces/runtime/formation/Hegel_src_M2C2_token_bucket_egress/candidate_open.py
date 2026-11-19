import math

_SCHEMA_FAMILY_TAG = 'rational_1d'
_CORE_LAW_EXPR = "(f['policer_rate'] / (0.2402671271 * f['policer_rate'] + 1.897467516))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['policer_rate'] / (0.2402671271 * f['policer_rate'] + 1.897467516))
    return 0 + (1) * (1) * latent
