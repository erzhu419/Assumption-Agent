import math

_SCHEMA_FAMILY_TAG = 'monod_plus_linear_2d'
_CORE_LAW_EXPR = "(((7.797820699 * f['caller_load']) / (2.4 + f['caller_load'])) + 0.5806936202 * f['vip_agents'])"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((7.797820699 * f['caller_load']) / (2.4 + f['caller_load'])) + 0.5806936202 * f['vip_agents'])
    return 3 + (-1) * (1.25) * latent
