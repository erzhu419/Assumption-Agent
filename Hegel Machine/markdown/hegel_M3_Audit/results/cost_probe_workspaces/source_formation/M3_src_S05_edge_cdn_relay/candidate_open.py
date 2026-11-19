import math

_SCHEMA_FAMILY_TAG = 'congestion_rollover_1d'
_CORE_LAW_EXPR = "(f['ingress_gbps'] / (0.2399828093 * f['ingress_gbps'] + 1.59730447 + 0.00102167036 * (f['ingress_gbps'] ** 2)))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['ingress_gbps'] / (0.2399828093 * f['ingress_gbps'] + 1.59730447 + 0.00102167036 * (f['ingress_gbps'] ** 2)))
    return 0 + (1) * (1) * latent
