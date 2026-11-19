import math

_SCHEMA_FAMILY_TAG = 'parallel_harmonic_flux_2d'
_CORE_LAW_EXPR = "((f['sample_delay_s'] * f['step_duty_v']) / (1e-05 * f['sample_delay_s'] + 1.661252293 * f['step_duty_v']))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.6}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['sample_delay_s'] * f['step_duty_v']) / (1e-05 * f['sample_delay_s'] + 1.661252293 * f['step_duty_v']))
    return 2.6 + (-1) * (1.25) * latent
