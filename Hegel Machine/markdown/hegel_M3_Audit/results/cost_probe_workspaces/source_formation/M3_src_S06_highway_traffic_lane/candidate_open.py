import math

_SCHEMA_FAMILY_TAG = 'congestion_rollover_1d'
_CORE_LAW_EXPR = "(f['veh_density'] / (0.219344707 * f['veh_density'] + 1.504114202 + 0.001825407246 * (f['veh_density'] ** 2)))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['veh_density'] / (0.219344707 * f['veh_density'] + 1.504114202 + 0.001825407246 * (f['veh_density'] ** 2)))
    return 0 + (1) * (1) * latent
