import math

_SCHEMA_FAMILY_TAG = 'liebig_capped_min_2d'
_CORE_LAW_EXPR = "min(2.439522186 * f['cooling_mw'], 2.147407281 * f['feeder_mw'], 16.99947688)"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 3.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(2.439522186 * f['cooling_mw'], 2.147407281 * f['feeder_mw'], 16.99947688)
    return 3.2 + (-1) * (1.25) * latent
