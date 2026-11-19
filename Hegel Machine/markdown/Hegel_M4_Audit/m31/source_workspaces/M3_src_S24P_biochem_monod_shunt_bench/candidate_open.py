import math

_SCHEMA_FAMILY_TAG = 'monod_plus_linear_2d'
_CORE_LAW_EXPR = "(((7.402498081 * f['substrate_mm']) / (2.4 + f['substrate_mm'])) + 0.5495296025 * f['bypass_lpm'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((7.402498081 * f['substrate_mm']) / (2.4 + f['substrate_mm'])) + 0.5495296025 * f['bypass_lpm'])
    return 0 + (1) * (1) * latent
