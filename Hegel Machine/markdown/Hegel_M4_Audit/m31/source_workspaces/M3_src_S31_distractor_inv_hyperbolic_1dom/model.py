import math

_SCHEMA_FAMILY_TAG = 'inv_linear_decay_1d'
_CORE_LAW_EXPR = "(0.54921806 + 9.20422819 / f['dose_mg'])"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 2.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.54921806 + 9.20422819 / f['dose_mg'])
    return 2 + (-1) * (1.2) * latent
