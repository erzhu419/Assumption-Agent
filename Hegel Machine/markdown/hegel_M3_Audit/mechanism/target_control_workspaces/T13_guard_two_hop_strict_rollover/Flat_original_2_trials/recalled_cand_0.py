import math

_SCHEMA_FAMILY_TAG = 'sqrt_orifice_1d'
_CORE_LAW_EXPR = "(0.5101932887 + 0.5108972695 * math.sqrt(f['offered_gbps']))"
_LENS = {'sign': -1.0, 'scale': 1.35, 'offset': 4.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.5101932887 + 0.5108972695 * math.sqrt(f['offered_gbps']))
    return 4.2 + (-1) * (1.35) * latent
