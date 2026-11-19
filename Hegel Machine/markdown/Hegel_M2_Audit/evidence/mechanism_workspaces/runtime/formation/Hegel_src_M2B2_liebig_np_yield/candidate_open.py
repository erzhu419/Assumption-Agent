import math

_SCHEMA_FAMILY_TAG = 'liebig_min_2d'
_CORE_LAW_EXPR = "min(1.834941198 * f['nitrate_kg'], 3.063220659 * f['phosphate_kg'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(1.834941198 * f['nitrate_kg'], 3.063220659 * f['phosphate_kg'])
    return 0 + (1) * (1) * latent
