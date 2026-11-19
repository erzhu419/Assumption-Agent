import math

_SCHEMA_FAMILY_TAG = 'rational_monod_1d'
_CORE_LAW_EXPR = "(f['packet_rate'] / (0.2500111849 * f['packet_rate'] + 1.747403632))"
_LENS = {'sign': -1.0, 'scale': 1.5, 'offset': 5.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['packet_rate'] / (0.2500111849 * f['packet_rate'] + 1.747403632))
    return 5 + (-1) * (1.5) * latent
