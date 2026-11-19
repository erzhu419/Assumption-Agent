import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'
_CORE_LAW_EXPR = "(0.8497871851 * f['flux_step'] + 1.350255239 * f['flux_step'] * math.exp(-f['window_s'] / 1.6))"
_LENS = {'sign': -1.0, 'scale': 1.5, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8497871851 * f['flux_step'] + 1.350255239 * f['flux_step'] * math.exp(-f['window_s'] / 1.6))
    return 3 + (-1) * (1.5) * latent
