import math

_SCHEMA_FAMILY_TAG = 'haldane_shunt_2d'
_CORE_LAW_EXPR = "(((8.015661108 * f['light_klux']) / (2.2 + f['light_klux'] + 0.065 * (f['light_klux'] ** 2))) + 0.5499286298 * f['bleed_lpm'])"
_LENS = {'sign': -1.0, 'scale': 1.2000000000000002, 'offset': 3.2}

def fit(observations):
    return None

def predict(row):
    f = row['features']
    latent = (((8.015661108 * f['light_klux']) / (2.2 + f['light_klux'] + 0.065 * (f['light_klux'] ** 2))) + 0.5499286298 * f['bleed_lpm'])
    return 3.2 + (-1) * (1.2) * latent
