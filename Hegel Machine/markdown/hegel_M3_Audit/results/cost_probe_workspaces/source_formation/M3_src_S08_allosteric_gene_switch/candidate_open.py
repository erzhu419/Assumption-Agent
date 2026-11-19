import math

_SCHEMA_FAMILY_TAG = 'hill_cooperative_n2_1d'
_CORE_LAW_EXPR = "((f['inducer_um'] ** 2) / (0.2598403213 * (f['inducer_um'] ** 2) + 8.009368154))"
_LENS = {'sign': -1.0, 'scale': 1.2999999999999998, 'offset': 4.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['inducer_um'] ** 2) / (0.2598403213 * (f['inducer_um'] ** 2) + 8.009368154))
    return 4 + (-1) * (1.3) * latent
