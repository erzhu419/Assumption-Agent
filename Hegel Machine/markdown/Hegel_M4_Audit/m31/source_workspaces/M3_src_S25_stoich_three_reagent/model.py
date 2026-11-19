import math

_SCHEMA_FAMILY_TAG = 'tri_resource_liebig_min_3d'
_CORE_LAW_EXPR = "min(1.80067734 * f['reag_a'], 2.096775671 * f['reag_b'], 2.400132557 * f['reag_c'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(1.80067734 * f['reag_a'], 2.096775671 * f['reag_b'], 2.400132557 * f['reag_c'])
    return 0 + (1) * (1) * latent
