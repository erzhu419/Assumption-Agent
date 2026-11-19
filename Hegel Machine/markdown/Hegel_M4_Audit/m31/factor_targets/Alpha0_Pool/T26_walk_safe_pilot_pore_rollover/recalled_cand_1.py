import math

_SCHEMA_FAMILY_TAG = 'sqrt_orifice_1d'
_CORE_LAW_EXPR = "(1.014318029 + 0.1696532084 * math.sqrt(f['monomer_flow']))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 4.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.014318029 + 0.1696532084 * math.sqrt(f['monomer_flow']))
    return 4.5 + (-1) * (1.4) * latent
