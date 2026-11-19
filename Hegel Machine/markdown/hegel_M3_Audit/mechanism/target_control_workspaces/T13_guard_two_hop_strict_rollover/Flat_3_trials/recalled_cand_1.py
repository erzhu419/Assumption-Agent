import math

_SCHEMA_FAMILY_TAG = 'log_saturation_1d'
_CORE_LAW_EXPR = "(0.1385582719 + 0.8792660512 * math.log(1.0 + f['offered_gbps']))"
_LENS = {'sign': -1.0, 'scale': 1.35, 'offset': 4.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.1385582719 + 0.8792660512 * math.log(1.0 + f['offered_gbps']))
    return 4.2 + (-1) * (1.35) * latent
