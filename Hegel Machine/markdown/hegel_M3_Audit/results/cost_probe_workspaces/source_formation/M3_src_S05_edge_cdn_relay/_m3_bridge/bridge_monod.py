import math

_SCHEMA_FAMILY_TAG = 'rational_monod_1d'
_CORE_LAW_EXPR = "(f['ingress_gbps'] / (0.2667633346 * f['ingress_gbps'] + 1.477847548))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['ingress_gbps'] / (0.2667633346 * f['ingress_gbps'] + 1.477847548))
    return 0 + (1) * (1) * latent
