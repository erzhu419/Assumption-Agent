import math

_SCHEMA_FAMILY_TAG = 'log_saturation_1d'
_CORE_LAW_EXPR = "(0.832114338 + 0.3223182379 * math.log(1.0 + f['monomer_flow']))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 4.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.832114338 + 0.3223182379 * math.log(1.0 + f['monomer_flow']))
    return 4.5 + (-1) * (1.4) * latent
