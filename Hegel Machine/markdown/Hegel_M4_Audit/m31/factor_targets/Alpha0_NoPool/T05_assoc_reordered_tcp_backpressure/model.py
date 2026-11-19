import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_reset'
_CORE_LAW_EXPR = "(0.8618022979 * f['burst_gbps'] + 1.323403862 * f['burst_gbps'] * math.exp(-f['horizon_s'] / 1.7))"
_LENS = {'sign': -1.0, 'scale': 1.35, 'offset': 2.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8618022979 * f['burst_gbps'] + 1.323403862 * f['burst_gbps'] * math.exp(-f['horizon_s'] / 1.7))
    return 2.2 + (-1) * (1.35) * latent
