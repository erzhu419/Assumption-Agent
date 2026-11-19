import math

_SCHEMA_FAMILY_TAG = 'log_saturation_1d'
_CORE_LAW_EXPR = "(0.3983973166 + 1.951477918 * math.log(1.0 + f['baffle_cm']))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.3983973166 + 1.951477918 * math.log(1.0 + f['baffle_cm']))
    return 0 + (1) * (1) * latent
