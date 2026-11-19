import math

_SCHEMA_FAMILY_TAG = 'hill_cooperative_n2_1d'
_CORE_LAW_EXPR = "((f['monomer_flow'] ** 2) / (0.5599635814 * (f['monomer_flow'] ** 2) + 2.135302614))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 4.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['monomer_flow'] ** 2) / (0.5599635814 * (f['monomer_flow'] ** 2) + 2.135302614))
    return 4.5 + (-1) * (1.4) * latent
