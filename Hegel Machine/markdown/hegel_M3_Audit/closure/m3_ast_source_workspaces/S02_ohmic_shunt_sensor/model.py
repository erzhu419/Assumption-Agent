import math

_SCHEMA_FAMILY_TAG = 'linear_proportional_1d'
_CORE_LAW_EXPR = "(0.8006781659 + 1.149968909 * f['current_a'])"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 2.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8006781659 + 1.149968909 * f['current_a'])
    return 2 + (-1) * (1.2) * latent
