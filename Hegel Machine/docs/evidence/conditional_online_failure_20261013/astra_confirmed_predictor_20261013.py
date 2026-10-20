THRESHOLD = -0.17615625
COEFFICIENTS = (
    (-0.27472345791918984, -0.939181223916844, -0.46123736686538264),
    (-0.1260050795145252, 0.8153019571318197, 0.7065292517721277),
)


def represent(row):
    window = row.get('history', [])[-2:] + [row['features']]
    return [
        float(row['features']['c']),
        len(window),
        sum(float(u['x']) for u in window),
        sum(float(u['z']) ** 2 for u in window),
    ]


def predict_features(features):
    c, count, sum_x, sum_z2 = features
    intercept, ax, az2 = COEFFICIENTS[int(c < THRESHOLD)]
    return float(intercept * count + ax * sum_x + az2 * sum_z2)


def predict(row):
    return predict_features(represent(row))
