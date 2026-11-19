import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'
_CORE_LAW_EXPR = "(0.9461190957 * f['surge_load'] + 1.432404964 * f['surge_load'] * math.exp(-f['horizon_s'] / 1.48))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.9461190957 * f['surge_load'] + 1.432404964 * f['surge_load'] * math.exp(-f['horizon_s'] / 1.48))
    return 2 + (-1) * (1.25) * latent
