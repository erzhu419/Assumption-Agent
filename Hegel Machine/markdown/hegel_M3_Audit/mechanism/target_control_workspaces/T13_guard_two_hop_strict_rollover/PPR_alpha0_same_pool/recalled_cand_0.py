import math

_SCHEMA_FAMILY_TAG = 'congestion_rollover_1d'
_CORE_LAW_EXPR = "(f['offered_gbps'] / (0.2334126922 * f['offered_gbps'] + 1.5528826 + 0.002224992044 * (f['offered_gbps'] ** 2)))"
_LENS = {'sign': -1.0, 'scale': 1.35, 'offset': 4.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['offered_gbps'] / (0.2334126922 * f['offered_gbps'] + 1.5528826 + 0.002224992044 * (f['offered_gbps'] ** 2)))
    return 4.2 + (-1) * (1.35) * latent
