import math

_SCHEMA_FAMILY_TAG = 'monod_plus_linear_2d'
_CORE_LAW_EXPR = "(((8.494578776 * f['cooling_mw']) / (5.55 + f['cooling_mw'])) + 0.8550408739 * f['feeder_mw'])"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 3.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((8.494578776 * f['cooling_mw']) / (5.55 + f['cooling_mw'])) + 0.8550408739 * f['feeder_mw'])
    return 3.2 + (-1) * (1.25) * latent
