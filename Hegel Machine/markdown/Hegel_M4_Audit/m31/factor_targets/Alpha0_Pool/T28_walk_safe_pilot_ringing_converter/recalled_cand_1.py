import math

_SCHEMA_FAMILY_TAG = 'liebig_min_2d'
_CORE_LAW_EXPR = "min(8.057128327 * f['sample_delay_s'], 2.086753607 * f['step_duty_v'])"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.6}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(8.057128327 * f['sample_delay_s'], 2.086753607 * f['step_duty_v'])
    return 2.6 + (-1) * (1.25) * latent
