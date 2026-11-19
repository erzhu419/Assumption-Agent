import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'
_CORE_LAW_EXPR = "(1.048948077 * f['flux_step'] + 1.098405694 * f['flux_step'] * math.exp(-f['window_dt'] / 1.36))"
_LENS = {'sign': -1.0, 'scale': 1.5, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.048948077 * f['flux_step'] + 1.098405694 * f['flux_step'] * math.exp(-f['window_dt'] / 1.36))
    return 3 + (-1) * (1.5) * latent
