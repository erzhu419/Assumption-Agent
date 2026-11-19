import math

_SCHEMA_FAMILY_TAG = 'haldane_shunt_2d'
_CORE_LAW_EXPR = "(((8.495697741 * f['substrate_mm']) / (2.2 + f['substrate_mm'] + 0.065 * (f['substrate_mm'] ** 2))) + 0.6207940603 * f['bypass_lpm'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((8.495697741 * f['substrate_mm']) / (2.2 + f['substrate_mm'] + 0.065 * (f['substrate_mm'] ** 2))) + 0.6207940603 * f['bypass_lpm'])
    return 0 + (1) * (1) * latent
