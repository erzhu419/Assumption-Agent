_OPPOSITE = np.array([2, 3, 0, 1])


def _bridge(calibration):
    """Fit the shared trigonometric bridge law from axis and mixed sweeps."""
    raw = np.asarray(calibration['bridge_raw'])
    loads = np.asarray(calibration['true_channel_loads'])
    sweeps = calibration['single_channel_sweeps']
    axis_raw = np.asarray(sweeps['cmd_stress'])
    # The gross readout jumps are separated from the clean sweeps by >10 units.
    clean = np.max(np.abs(raw), axis=1) < 5.0
    axis_clean = np.abs(axis_raw) < 5.0
    labs = np.asarray(sweeps['lab_id'])[axis_clean]
    axis_x = np.asarray(sweeps['true_stress'])[axis_clean]
    axis_y = axis_raw[axis_clean]
    x, y = loads[clean], raw[clean]
    w = x[:, _OPPOSITE]

    def axis_design(freq):
        return np.column_stack((np.ones_like(axis_x), axis_x, np.sin(freq * axis_x)))

    def profile_axis(freq, coefficients=False):
        a = axis_design(freq[0])
        b = np.stack([np.linalg.lstsq(a[labs == j], axis_y[labs == j], rcond=None)[0]
                      for j in range(4)])
        return b if coefficients else (a * b[labs]).sum(axis=1) - axis_y

    f = least_squares(profile_axis, [2.5], bounds=(1.0, 4.0)).x
    axis = profile_axis(f, coefficients=True)
    offset, gain, sine_amplitude = axis.T
    residual = y - offset - gain*x - sine_amplitude*np.sin(f[0]*x)

    def profile_cross(freq, coefficients=False):
        a = np.column_stack(((gain*w).ravel(), (gain*(1-np.cos(freq[0]*w))).ravel()))
        b = np.linalg.lstsq(a, residual.ravel(), rcond=None)[0]
        return b if coefficients else a @ b - residual.ravel()

    g = least_squares(profile_cross, [2.0], bounds=(0.5, 4.0)).x
    cross, cosine = profile_cross(g, coefficients=True)
    return offset, gain, sine_amplitude / gain, f[0], cross, cosine, g[0]


def _loads(raw, calibration):
    offset, gain, sine, f, cross, cosine, g = calibration
    y = (np.asarray(raw) - offset) / gain
    # Newton inversion treats the two opposite-channel pairs independently.
    x = y.copy()
    for _ in range(20):
        w = x[..., _OPPOSITE]
        error = x + sine*np.sin(f*x) + cross*w + cosine*(1-np.cos(g*w)) - y
        diag = 1 + sine*f*np.cos(f*x)
        off = cross + cosine*g*np.sin(g*w)
        determinant = diag*diag[..., _OPPOSITE] - off*off[..., _OPPOSITE]
        step = (diag[..., _OPPOSITE]*error - off*error[..., _OPPOSITE]) / determinant
        x -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return x


