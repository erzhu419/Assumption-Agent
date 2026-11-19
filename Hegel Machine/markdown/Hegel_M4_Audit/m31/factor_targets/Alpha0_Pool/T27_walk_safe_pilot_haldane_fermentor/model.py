import math

_SCHEMA_FAMILY_TAG = 'haldane_shunt_2d'
_CORE_LAW_EXPR = "(((7.99947651 * f['carbon_mm']) / (2.2 + f['carbon_mm'] + 0.065 * (f['carbon_mm'] ** 2))) + 0.5800823854 * f['bleed_rate'])"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((7.99947651 * f['carbon_mm']) / (2.2 + f['carbon_mm'] + 0.065 * (f['carbon_mm'] ** 2))) + 0.5800823854 * f['bleed_rate'])
    return 3 + (-1) * (1.2) * latent
