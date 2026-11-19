import math

_SCHEMA_FAMILY_TAG = 'gaussian_weibull_relax_2d'
_CORE_LAW_EXPR = "(0.8828443099 * f['burst_gbps'] + 0.9614688896 * f['burst_gbps'] * math.exp(-(f['horizon_s'] ** 2) / 4.6))"
_LENS = {'sign': -1.0, 'scale': 1.35, 'offset': 2.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8828443099 * f['burst_gbps'] + 0.9614688896 * f['burst_gbps'] * math.exp(-(f['horizon_s'] ** 2) / 4.6))
    return 2.2 + (-1) * (1.35) * latent
