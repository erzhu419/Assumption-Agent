import math

_SCHEMA_FAMILY_TAG = 'rational_monod_1d'
_CORE_LAW_EXPR = "(f['feed_conc'] / (0.2802075287 * f['feed_conc'] + 1.599035791))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['feed_conc'] / (0.2802075287 * f['feed_conc'] + 1.599035791))
    return 0 + (1) * (1) * latent
