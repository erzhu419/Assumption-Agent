import math

_SCHEMA_FAMILY_TAG = 'liebig_capped_min_2d'
_CORE_LAW_EXPR = "min(0.9580028266 * f['horizon_s'], 0.9733578696 * f['surge_load'], 9.613634728)"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(0.9580028266 * f['horizon_s'], 0.9733578696 * f['surge_load'], 9.613634728)
    return 2 + (-1) * (1.25) * latent
