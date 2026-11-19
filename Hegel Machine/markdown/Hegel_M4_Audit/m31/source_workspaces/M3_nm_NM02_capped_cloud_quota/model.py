import math

_SCHEMA_FAMILY_TAG = 'liebig_capped_min_2d'
_CORE_LAW_EXPR = "min(2.150010159 * f['vcpu_cores'], 2.501088442 * f['iops_k'], 17.19740343)"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(2.150010159 * f['vcpu_cores'], 2.501088442 * f['iops_k'], 17.19740343)
    return 0 + (1) * (1) * latent
