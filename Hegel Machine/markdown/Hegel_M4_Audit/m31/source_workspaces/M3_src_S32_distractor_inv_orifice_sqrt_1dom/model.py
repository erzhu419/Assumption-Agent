import math

_SCHEMA_FAMILY_TAG = 'sqrt_orifice_1d'
_CORE_LAW_EXPR = "(0.6982531619 + 1.850198587 * math.sqrt(f['dp_kpa']))"
_LENS = {'sign': -1.0, 'scale': 1.1, 'offset': 1.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.6982531619 + 1.850198587 * math.sqrt(f['dp_kpa']))
    return 1.5 + (-1) * (1.1) * latent
