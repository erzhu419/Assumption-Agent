import math

_SCHEMA_FAMILY_TAG = 'underdamped_ringing_2d'
_CORE_LAW_EXPR = "(1.600016632 * f['power_w'] * (1.0 - math.exp(-f['elapsed_s'] / 2.2) * math.cos(1.68 * f['elapsed_s'])))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.600016632 * f['power_w'] * (1.0 - math.exp(-f['elapsed_s'] / 2.2) * math.cos(1.68 * f['elapsed_s'])))
    return 0 + (1) * (1) * latent
