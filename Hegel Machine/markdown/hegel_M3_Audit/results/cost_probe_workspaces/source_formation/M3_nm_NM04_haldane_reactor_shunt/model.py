import math

_SCHEMA_FAMILY_TAG = 'haldane_shunt_2d'
_CORE_LAW_EXPR = "(((8.500075013 * f['feed_conc']) / (2.2 + f['feed_conc'] + 0.065 * (f['feed_conc'] ** 2))) + 0.6201342709 * f['bypass_stream'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((8.500075013 * f['feed_conc']) / (2.2 + f['feed_conc'] + 0.065 * (f['feed_conc'] ** 2))) + 0.6201342709 * f['bypass_stream'])
    return 0 + (1) * (1) * latent
