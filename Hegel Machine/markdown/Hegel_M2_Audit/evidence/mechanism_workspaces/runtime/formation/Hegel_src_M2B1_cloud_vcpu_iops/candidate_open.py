import math

_SCHEMA_FAMILY_TAG = 'liebig_min_2d'
_CORE_LAW_EXPR = "min(1.864745254 * f['vcpu_share'], 2.375222754 * f['iops_quota'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(1.864745254 * f['vcpu_share'], 2.375222754 * f['iops_quota'])
    return 0 + (1) * (1) * latent
