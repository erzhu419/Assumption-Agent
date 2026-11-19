import math

_SCHEMA_FAMILY_TAG = 'liebig_min_2d'
_CORE_LAW_EXPR = "min(1.333168964 * f['vcpu_cores'], 2.500642769 * f['iops_k'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(1.333168964 * f['vcpu_cores'], 2.500642769 * f['iops_k'])
    return 0 + (1) * (1) * latent
