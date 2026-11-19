import math

_SCHEMA_FAMILY_TAG = 'liebig_capped_min_2d'
_CORE_LAW_EXPR = "min(2.55636158 * f['chiller_kw'], 2.204797292 * f['pdu_kw'], 16.99876866)"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(2.55636158 * f['chiller_kw'], 2.204797292 * f['pdu_kw'], 16.99876866)
    return 3 + (-1) * (1.25) * latent
