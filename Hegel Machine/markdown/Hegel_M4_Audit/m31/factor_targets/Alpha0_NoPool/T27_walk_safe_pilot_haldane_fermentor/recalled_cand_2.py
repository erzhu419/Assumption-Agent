import math

_SCHEMA_FAMILY_TAG = 'parallel_harmonic_flux_2d'
_CORE_LAW_EXPR = "((f['bleed_rate'] * f['carbon_mm']) / (1e-05 * f['bleed_rate'] + 0.3789412487 * f['carbon_mm']))"
_LENS = {'sign': -1.0, 'scale': 1.2, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['bleed_rate'] * f['carbon_mm']) / (1e-05 * f['bleed_rate'] + 0.3789412487 * f['carbon_mm']))
    return 3 + (-1) * (1.2) * latent
