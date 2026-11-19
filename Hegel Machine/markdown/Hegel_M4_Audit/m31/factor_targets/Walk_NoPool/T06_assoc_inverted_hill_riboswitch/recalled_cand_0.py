import math

_SCHEMA_FAMILY_TAG = 'hill_cooperative_n2_1d'
_CORE_LAW_EXPR = "((f['ligand_nm'] ** 2) / (0.2906612736 * (f['ligand_nm'] ** 2) + 8.494874259))"
_LENS = {'sign': -1.0, 'scale': 1.4, 'offset': 4.5}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['ligand_nm'] ** 2) / (0.2906612736 * (f['ligand_nm'] ** 2) + 8.494874259))
    return 4.5 + (-1) * (1.4) * latent
