import math

_SCHEMA_FAMILY_TAG = 'parallel_harmonic_flux_2d'
_CORE_LAW_EXPR = "((f['cond_pad'] * f['cond_fin']) / (0.4200560046 * f['cond_pad'] + 0.5797427631 * f['cond_fin']))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['cond_pad'] * f['cond_fin']) / (0.4200560046 * f['cond_pad'] + 0.5797427631 * f['cond_fin']))
    return 0 + (1) * (1) * latent
