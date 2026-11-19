import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_state_memory'
_CORE_LAW_EXPR = "(0.8820212334 * f['pulse_a'] + 1.383055622 * f['pulse_a'] * math.exp(-f['rest_s'] / 1.7) + 0.9495256727 * f['prior_overpot'] * math.exp(-f['rest_s'] / 1.7))"
_LENS = {'sign': -1.0, 'scale': 1.3, 'offset': 2.8}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8820212334 * f['pulse_a'] + 1.383055622 * f['pulse_a'] * math.exp(-f['rest_s'] / 1.7) + 0.9495256727 * f['prior_overpot'] * math.exp(-f['rest_s'] / 1.7))
    return 2.8 + (-1) * (1.3) * latent
