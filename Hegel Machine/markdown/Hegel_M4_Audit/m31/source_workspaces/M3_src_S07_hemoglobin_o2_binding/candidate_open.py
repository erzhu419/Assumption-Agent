import math

_SCHEMA_FAMILY_TAG = 'hill_cooperative_n2_1d'
_CORE_LAW_EXPR = "((f['po2_torr'] ** 2) / (0.319916025 * (f['po2_torr'] ** 2) + 9.494819274))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['po2_torr'] ** 2) / (0.319916025 * (f['po2_torr'] ** 2) + 9.494819274))
    return 0 + (1) * (1) * latent
