import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_state_memory'
_CORE_LAW_EXPR = "(0.8957032784 * f['flux_step'] + 1.2936802 * f['flux_step'] * math.exp(-f['window_dt'] / 1.84) + 0.9891704538 * f['init_current'] * math.exp(-f['window_dt'] / 1.84))"
_LENS = {'sign': -1.0, 'scale': 1.0, 'offset': 1.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8957032784 * f['flux_step'] + 1.2936802 * f['flux_step'] * math.exp(-f['window_dt'] / 1.84) + 0.9891704538 * f['init_current'] * math.exp(-f['window_dt'] / 1.84))
    return 1.5 + (-1) * (1) * latent
