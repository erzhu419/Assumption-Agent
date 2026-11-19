import math

_SCHEMA_FAMILY_TAG = 'underdamped_ringing_2d'
_CORE_LAW_EXPR = "(1.553398601 * f['step_v'] * (1.0 - math.exp(-f['probe_s'] / 2.2) * math.cos(1.68 * f['probe_s'])))"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 2.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.553398601 * f['step_v'] * (1.0 - math.exp(-f['probe_s'] / 2.2) * math.cos(1.68 * f['probe_s'])))
    return 2.5 + (-1) * (1.2) * latent
