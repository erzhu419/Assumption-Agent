import math

_SCHEMA_FAMILY_TAG = 'liebig_capped_min_2d'
_CORE_LAW_EXPR = "min(2.100014432 * f['carbon_feed'], 2.401877991 * f['aeration_rate'], 17.50134966)"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(2.100014432 * f['carbon_feed'], 2.401877991 * f['aeration_rate'], 17.50134966)
    return 0 + (1) * (1) * latent
