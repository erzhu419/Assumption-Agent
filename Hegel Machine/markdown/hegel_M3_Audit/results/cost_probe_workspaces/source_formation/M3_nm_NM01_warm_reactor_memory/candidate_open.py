import math

_SCHEMA_FAMILY_TAG = 'conjugate_relaxation_state_memory'
_CORE_LAW_EXPR = "(0.8999843699 * f['pressure_step'] + 1.401150824 * f['pressure_step'] * math.exp(-f['settle_s'] / 1.8) + 1.001525948 * f['residual_extent'] * math.exp(-f['settle_s'] / 1.8))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (0.8999843699 * f['pressure_step'] + 1.401150824 * f['pressure_step'] * math.exp(-f['settle_s'] / 1.8) + 1.001525948 * f['residual_extent'] * math.exp(-f['settle_s'] / 1.8))
    return 0 + (1) * (1) * latent
