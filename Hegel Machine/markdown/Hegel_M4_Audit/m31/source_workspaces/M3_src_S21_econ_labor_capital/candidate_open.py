import math

_SCHEMA_FAMILY_TAG = 'cobb_douglas_sqrt_2d'
_CORE_LAW_EXPR = "(2.29993727 * math.sqrt(f['labor_idx'] * f['capital_idx']))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (2.29993727 * math.sqrt(f['labor_idx'] * f['capital_idx']))
    return 0 + (1) * (1) * latent
