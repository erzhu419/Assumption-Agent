import math

_SCHEMA_FAMILY_TAG = 'overdamped_charging_2d'
_CORE_LAW_EXPR = "(0.2557785873 * f['carbon_mm'] * (1.0 - math.exp(-f['bleed_rate'] / 0.4475)))"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.2557785873 * f['carbon_mm'] * (1.0 - math.exp(-f['bleed_rate'] / 0.4475)))
    return 3 + (-1) * (1.2) * latent
