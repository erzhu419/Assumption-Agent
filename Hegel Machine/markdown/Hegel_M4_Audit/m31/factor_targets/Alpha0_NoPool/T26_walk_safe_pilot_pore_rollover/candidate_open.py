import math

_SCHEMA_FAMILY_TAG = 'congestion_rollover_1d'
_CORE_LAW_EXPR = "(f['monomer_flow'] / (0.243823178 * f['monomer_flow'] + 1.65192715 + 0.01199234676 * (f['monomer_flow'] ** 2)))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 4.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['monomer_flow'] / (0.243823178 * f['monomer_flow'] + 1.65192715 + 0.01199234676 * (f['monomer_flow'] ** 2)))
    return 4.5 + (-1) * (1.4) * latent
