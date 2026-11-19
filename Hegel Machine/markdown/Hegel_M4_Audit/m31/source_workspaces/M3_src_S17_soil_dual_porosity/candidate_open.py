import math

_SCHEMA_FAMILY_TAG = 'bimodal_relaxation_2mode'
_CORE_LAW_EXPR = "(0.7199420884 * f['head_m'] + 2.749100132 * f['head_m'] * math.exp(-f['drain_hr'] / 0.55) + 0.920260722 * f['head_m'] * math.exp(-f['drain_hr'] / 6))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.7199420884 * f['head_m'] + 2.749100132 * f['head_m'] * math.exp(-f['drain_hr'] / 0.55) + 0.920260722 * f['head_m'] * math.exp(-f['drain_hr'] / 6))
    return 0 + (1) * (1) * latent
