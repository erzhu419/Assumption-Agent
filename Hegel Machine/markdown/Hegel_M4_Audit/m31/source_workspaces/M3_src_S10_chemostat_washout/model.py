import math

_SCHEMA_FAMILY_TAG = 'exp_washout_decay_1d'
_CORE_LAW_EXPR = "(0.5995571862 + 4.493702406 * math.exp(-f['dilution_hr'] / 4.612))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.5995571862 + 4.493702406 * math.exp(-f['dilution_hr'] / 4.612))
    return 2.2 + (-1) * (1.25) * latent
