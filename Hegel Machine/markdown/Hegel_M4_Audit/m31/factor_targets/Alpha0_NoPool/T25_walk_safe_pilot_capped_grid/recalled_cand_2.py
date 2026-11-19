import math

_SCHEMA_FAMILY_TAG = 'parallel_harmonic_flux_2d'
_CORE_LAW_EXPR = "((f['cooling_mw'] * f['feeder_mw']) / (0.4013564683 * f['cooling_mw'] + 0.3120762674 * f['feeder_mw']))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 3.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['cooling_mw'] * f['feeder_mw']) / (0.4013564683 * f['cooling_mw'] + 0.3120762674 * f['feeder_mw']))
    return 3.2 + (-1) * (1.25) * latent
