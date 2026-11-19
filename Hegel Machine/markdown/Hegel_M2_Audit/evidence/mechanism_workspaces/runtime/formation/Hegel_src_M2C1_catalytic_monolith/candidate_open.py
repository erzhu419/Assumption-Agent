import math

_SCHEMA_FAMILY_TAG = 'rational_1d'
_CORE_LAW_EXPR = "(f['reactant_feed'] / (0.2801607569 * f['reactant_feed'] + 1.599842651))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (f['reactant_feed'] / (0.2801607569 * f['reactant_feed'] + 1.599842651))
    return 0 + (1) * (1) * latent
