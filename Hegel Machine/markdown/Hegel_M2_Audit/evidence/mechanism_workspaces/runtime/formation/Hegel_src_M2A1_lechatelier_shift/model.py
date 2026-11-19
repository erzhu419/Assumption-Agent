import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'
_CORE_LAW_EXPR = "(0.830519921 * f['pressure_step'] + 1.513067239 * f['pressure_step'] * math.exp(-f['settle_time'] / 1.72))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.830519921 * f['pressure_step'] + 1.513067239 * f['pressure_step'] * math.exp(-f['settle_time'] / 1.72))
    return 0 + (1) * (1) * latent
