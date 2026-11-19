import math

_SCHEMA_FAMILY_TAG = 'liebig_min_2d'
_CORE_LAW_EXPR = "min(51.66747885 * f['bleed_rate'], 1.460564532 * f['carbon_mm'])"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(51.66747885 * f['bleed_rate'], 1.460564532 * f['carbon_mm'])
    return 3 + (-1) * (1.2) * latent
