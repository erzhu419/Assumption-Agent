import math

_SCHEMA_FAMILY_TAG = 'overdamped_charging_2d'
_CORE_LAW_EXPR = "(4.392757802 * f['step_duty_v'] * (1.0 - math.exp(-f['sample_delay_s'] / 4.2525)))"
_LENS = {'sign': -1.0, 'scale': 1.25, 'offset': 2.6}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (4.392757802 * f['step_duty_v'] * (1.0 - math.exp(-f['sample_delay_s'] / 4.2525)))
    return 2.6 + (-1) * (1.25) * latent
