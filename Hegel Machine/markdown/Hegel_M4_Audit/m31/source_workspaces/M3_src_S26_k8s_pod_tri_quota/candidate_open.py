import math

_SCHEMA_FAMILY_TAG = 'tri_resource_liebig_min_3d'
_CORE_LAW_EXPR = "min(2.002258426 * f['cpu_lim'], 1.901269478 * f['mem_lim'], 2.298969923 * f['nvme_lim'])"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = min(2.002258426 * f['cpu_lim'], 1.901269478 * f['mem_lim'], 2.298969923 * f['nvme_lim'])
    return 0 + (1) * (1) * latent
