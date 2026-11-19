import math

_SCHEMA_FAMILY_TAG = 'congestion_rollover_1d'
_CORE_LAW_EXPR = "(f['feed_conc'] / (0.2567892447 * f['feed_conc'] + 1.613227342 + 0.01265553713 * (f['feed_conc'] ** 2)))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['feed_conc'] / (0.2567892447 * f['feed_conc'] + 1.613227342 + 0.01265553713 * (f['feed_conc'] ** 2)))
    return 0 + (1) * (1) * latent
