import math

_SCHEMA_FAMILY_TAG = 'inv_quad_crest_1d'
_CORE_LAW_EXPR = "(6.195838302 + -8.789158691 / f['u'] + 13.37804869 / (f['u'] ** 2))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (6.195838302 + -8.789158691 / f['u'] + 13.37804869 / (f['u'] ** 2))
    return 0 + (1) * (1) * latent
