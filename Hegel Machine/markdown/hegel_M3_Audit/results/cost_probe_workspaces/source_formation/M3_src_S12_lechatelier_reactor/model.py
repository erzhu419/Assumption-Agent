import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'
_CORE_LAW_EXPR = "(0.9201251685 * f['pressure_step'] + 1.449667833 * f['pressure_step'] * math.exp(-f['settle_s'] / 1.8))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.9201251685 * f['pressure_step'] + 1.449667833 * f['pressure_step'] * math.exp(-f['settle_s'] / 1.8))
    return 0 + (1) * (1) * latent
