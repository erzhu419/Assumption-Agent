import math

_SCHEMA_FAMILY_TAG = 'rational_monod_1d'
_CORE_LAW_EXPR = "(f['bypass_stream'] / (0.1600962336 * f['bypass_stream'] + 0.03053587347))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['bypass_stream'] / (0.1600962336 * f['bypass_stream'] + 0.03053587347))
    return 0 + (1) * (1) * latent
