import math

_SCHEMA_FAMILY_TAG = 'parallel_harmonic_flux_2d'
_CORE_LAW_EXPR = "((f['stage1_cap'] * f['stage2_cap']) / (0.3800191755 * f['stage1_cap'] + 0.6498749899 * f['stage2_cap']))"
_LENS = {'sign': -1.0, 'scale': 1.1, 'offset': 2.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['stage1_cap'] * f['stage2_cap']) / (0.3800191755 * f['stage1_cap'] + 0.6498749899 * f['stage2_cap']))
    return 2 + (-1) * (1.1) * latent
