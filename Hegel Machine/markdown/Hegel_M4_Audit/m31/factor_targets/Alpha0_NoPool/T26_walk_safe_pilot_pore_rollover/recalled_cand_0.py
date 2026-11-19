import math

_SCHEMA_FAMILY_TAG = 'rational_monod_1d'
_CORE_LAW_EXPR = "(f['monomer_flow'] / (0.3490176587 * f['monomer_flow'] + 1.454593298))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 4.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['monomer_flow'] / (0.3490176587 * f['monomer_flow'] + 1.454593298))
    return 4.5 + (-1) * (1.4) * latent
