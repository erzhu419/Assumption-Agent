import math

_SCHEMA_FAMILY_TAG = 'cobb_douglas_sqrt_2d'
_CORE_LAW_EXPR = "(2.550101297 * math.sqrt(f['gpu_count'] * f['nvlink_bw']))"
_LENS = {'sign': 1.0, 'scale': 1.0, 'offset': 0.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (2.550101297 * math.sqrt(f['gpu_count'] * f['nvlink_bw']))
    return 0 + (1) * (1) * latent
