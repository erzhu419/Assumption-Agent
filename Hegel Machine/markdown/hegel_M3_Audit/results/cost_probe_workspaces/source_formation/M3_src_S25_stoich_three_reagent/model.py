import math

_SCHEMA_FAMILY_TAG = 'tri_resource_liebig_min_3d'
_CORE_LAW_EXPR = "min(1.799753999 * f['reag_a'], 2.097429143 * f['reag_b'], 2.39966537 * f['reag_c'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(1.799753999 * f['reag_a'], 2.097429143 * f['reag_b'], 2.39966537 * f['reag_c'])
    return 0 + (1) * (1) * latent
