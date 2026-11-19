import math

_SCHEMA_FAMILY_TAG = 'underdamped_ringing_2d'
_CORE_LAW_EXPR = "(1.521889146 * f['step_duty_v'] * (1.0 - math.exp(-f['sample_delay_s'] / 2.2) * math.cos(1.68 * f['sample_delay_s'])))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.6}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (1.521889146 * f['step_duty_v'] * (1.0 - math.exp(-f['sample_delay_s'] / 2.2) * math.cos(1.68 * f['sample_delay_s'])))
    return 2.6 + (-1) * (1.25) * latent
