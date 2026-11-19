import math

_SCHEMA_FAMILY_TAG = 'liebig_min_2d'
_CORE_LAW_EXPR = "min(1.850778072 * f['nitrate_kg'], 2.650395617 * f['phosphate_kg'])"
_LENS = {'sign': -1.0, 'scale': 1.2000000000000002, 'offset': 3.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(1.850778072 * f['nitrate_kg'], 2.650395617 * f['phosphate_kg'])
    return 3.5 + (-1) * (1.2) * latent
