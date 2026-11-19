import math

_SCHEMA_FAMILY_TAG = 'damped_driven_with_leak_3d'
_CORE_LAW_EXPR = "(1.350421803 * f['rain_mm'] * (1.0 - math.exp(-f['lag_hr'] / 1.6)) + 0.5499958084 * f['baseflow'])"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 2.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.350421803 * f['rain_mm'] * (1.0 - math.exp(-f['lag_hr'] / 1.6)) + 0.5499958084 * f['baseflow'])
    return 2.5 + (-1) * (1.2) * latent
