import math

_SCHEMA_FAMILY_TAG = 'langmuir_product_2d'
_CORE_LAW_EXPR = "((f['prey_dens'] * f['habitat_ha']) / (1.797078606 + 0.3497490629 * f['prey_dens'] + 0.4505062861 * f['habitat_ha']))"
_LENS = {'sign': -1.0, 'scale': 1.3, 'offset': 3.0}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = ((f['prey_dens'] * f['habitat_ha']) / (1.797078606 + 0.3497490629 * f['prey_dens'] + 0.4505062861 * f['habitat_ha']))
    return 3 + (-1) * (1.3) * latent
