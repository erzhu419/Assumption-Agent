import math

_SCHEMA_FAMILY_TAG = 'overdamped_charging_2d'
_CORE_LAW_EXPR = "(1.418797315 * f['cooling_mw'] * (1.0 - math.exp(-f['feeder_mw'] / 0.8875)))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 3.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.418797315 * f['cooling_mw'] * (1.0 - math.exp(-f['feeder_mw'] / 0.8875)))
    return 3.2 + (-1) * (1.25) * latent
