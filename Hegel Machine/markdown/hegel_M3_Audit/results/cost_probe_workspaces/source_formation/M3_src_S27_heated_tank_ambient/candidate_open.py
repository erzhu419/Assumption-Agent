import math

_SCHEMA_FAMILY_TAG = 'damped_driven_with_leak_3d'
_CORE_LAW_EXPR = "(1.49989739 * f['steam_kw'] * (1.0 - math.exp(-f['soak_s'] / 1.8)) + 0.6501731222 * f['ambient_c'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.49989739 * f['steam_kw'] * (1.0 - math.exp(-f['soak_s'] / 1.8)) + 0.6501731222 * f['ambient_c'])
    return 0 + (1) * (1) * latent
