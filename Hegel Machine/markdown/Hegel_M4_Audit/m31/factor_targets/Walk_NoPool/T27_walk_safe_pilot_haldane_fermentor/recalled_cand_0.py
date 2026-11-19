import math

_SCHEMA_FAMILY_TAG = 'monod_plus_linear_2d'
_CORE_LAW_EXPR = "(((6.549192796 * f['carbon_mm']) / (1.7 + f['carbon_mm'])) + 0.6203933488 * f['bleed_rate'])"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((6.549192796 * f['carbon_mm']) / (1.7 + f['carbon_mm'])) + 0.6203933488 * f['bleed_rate'])
    return 3 + (-1) * (1.2) * latent
