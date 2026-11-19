import math

_SCHEMA_FAMILY_TAG = 'bimodal_relaxation_2mode'
_CORE_LAW_EXPR = "(0.6499235844 * f['strain_pct'] + 2.849083603 * f['strain_pct'] * math.exp(-f['hold_s'] / 0.55) + 0.950039248 * f['strain_pct'] * math.exp(-f['hold_s'] / 6))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.6499235844 * f['strain_pct'] + 2.849083603 * f['strain_pct'] * math.exp(-f['hold_s'] / 0.55) + 0.950039248 * f['strain_pct'] * math.exp(-f['hold_s'] / 6))
    return 0 + (1) * (1) * latent
