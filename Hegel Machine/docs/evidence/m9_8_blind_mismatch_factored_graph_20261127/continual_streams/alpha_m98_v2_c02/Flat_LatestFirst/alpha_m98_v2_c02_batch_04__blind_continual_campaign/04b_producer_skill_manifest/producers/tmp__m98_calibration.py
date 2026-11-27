def _stress_calibration(cal):
    load = cal['true_channel_loads']
    raw = cal['bridge_raw']
    nxt = np.array([1, 2, 3, 0])
    cols = np.arange(4)
    base = np.column_stack((np.ones(len(load)), load))
    cubic = load**3
    cross = load * load[:, nxt]

    def fit(mask):
        return np.array([np.linalg.lstsq(
            np.column_stack((base, cubic[:, j], cross[:, j]))[mask],
            raw[mask, j], rcond=None)[0] for j in cols]).T

    # The mixed-load measurements identify all terms independently of the axis sweeps.
    coef = fit(np.count_nonzero(load, axis=1) > 1)
    err = np.linalg.norm(base @ coef[:5] + cubic * coef[5] + cross * coef[6] - raw, axis=1)
    median = np.median(err)
    mad = 1.4826 * np.median(np.abs(err - median))
    clean = err <= median + 10 * max(mad, 1e-8)
    bias, linear, cubic_gain, cross_gain = coef[0], coef[1:5], coef[5], coef[6]
    coef = fit(clean)
    bias, linear, cubic_gain, cross_gain = coef[0], coef[1:5], coef[5], coef[6]

    def channel_loads(bridge):
        shape = bridge.shape
        y = bridge.reshape(-1, 4) - bias
        x = np.linalg.solve(linear.T, y.T).T
        for _ in range(15):
            z = x[:, nxt]
            r = x @ linear + x**3 * cubic_gain + x * z * cross_gain - y
            if np.max(np.abs(r)) < 1e-12:
                break
            jac = np.broadcast_to(linear.T, (len(x), 4, 4)).copy()
            jac[:, cols, cols] += 3 * x**2 * cubic_gain + z * cross_gain
            jac[:, cols, nxt] += x * cross_gain
            x -= np.linalg.solve(jac, r[..., None])[..., 0]
        return x.reshape(shape)

    def stress(bridge):
        return channel_loads(bridge).sum(axis=-1)

    return stress
