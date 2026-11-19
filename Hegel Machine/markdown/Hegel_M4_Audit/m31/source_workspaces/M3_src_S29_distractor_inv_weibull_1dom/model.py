import math

_SCHEMA_FAMILY_TAG = 'gaussian_weibull_relax_2d'
_CORE_LAW_EXPR = "(0.7499658374 * f['spl_pa'] + 1.450107074 * f['spl_pa'] * math.exp(-(f['decay_s'] ** 2) / 3.2))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 2.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.7499658374 * f['spl_pa'] + 1.450107074 * f['spl_pa'] * math.exp(-(f['decay_s'] ** 2) / 3.2))
    return 2.5 + (-1) * (1.4) * latent
