import math

_SCHEMA_FAMILY_TAG = 'damped_driven_with_leak_3d'
_CORE_LAW_EXPR = "(1.248598147 * f['pulse_a'] * (1.0 - math.exp(-f['rest_s'] / 0.75)) + 0.08573705113 * f['prior_overpot'])"
_LENS = {'sign': -1.0, 'scale': 1.3, 'offset': 2.8}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.248598147 * f['pulse_a'] * (1.0 - math.exp(-f['rest_s'] / 0.75)) + 0.08573705113 * f['prior_overpot'])
    return 2.8 + (-1) * (1.3) * latent
