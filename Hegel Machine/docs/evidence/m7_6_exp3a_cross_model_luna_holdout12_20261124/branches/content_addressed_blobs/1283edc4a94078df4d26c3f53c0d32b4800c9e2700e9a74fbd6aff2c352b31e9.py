"""Two-absorber localization for calibrated ring thermoacoustic measurements."""
import numpy as np
from scipy.signal import correlate
from scipy.optimize import differential_evolution


def reconstruct_two_absorbers_mm(scan: dict) -> dict:
    signals = np.asarray(scan['signals'], dtype=float)
    sensors = np.asarray(scan['sensor_xy_mm'], dtype=float)
    fs = float(scan['fs_hz'])
    n, m = signals.shape
    if m != len(sensors):
        raise ValueError('signals must have one column per sensor')
    # The bipolar N-wave is proportional to -t exp(-t²/(2 tau²)); its
    # matched filter is insensitive to channel gain and fitted polarity.
    z = np.arange(-12, 13, dtype=float)
    template = -z * np.exp(-0.5 * (z / 3.0) ** 2)
    template /= np.linalg.norm(template)
    corr = np.column_stack([
        correlate(signals[:, j], template, mode='same', method='fft')
        for j in range(m)
    ])
    channel_norm = np.sqrt(np.sum(signals * signals, axis=0)) + 1e-12
    corr = corr / channel_norm[None, :]
    ix = np.arange(m)

    def arrival_field(x, y, c):
        delay = np.linalg.norm(sensors - np.array([x, y]), axis=1) * fs / (c * 1000.0)
        lo = np.floor(delay).astype(int)
        frac = delay - lo
        lo = np.clip(lo, 0, n - 1)
        hi = np.minimum(lo + 1, n - 1)
        return corr[lo, ix] * (1-frac) + corr[hi, ix] * frac

    def objective(v):
        x1, y1, x2, y2, c = v
        sep = np.hypot(x1-x2, y1-y2)
        if sep < 8.0:
            return 100.0 + 100.0*(8.0-sep)**2
        # Each channel has independent unknown polarity and gain; absolute
        # matched-filter evidence therefore preserves timing but not amplitude.
        return -np.sum(np.abs(arrival_field(x1,y1,c)) + np.abs(arrival_field(x2,y2,c)))

    result = differential_evolution(
        objective, [(-25,25),(-25,25),(-25,25),(-25,25),(1435,1545)],
        seed=3, popsize=18, maxiter=350, tol=1e-8, polish=True,
        updating='immediate', workers=1
    )
    pts = np.asarray(result.x[:4]).reshape(2,2)
    pts = pts[np.argsort(pts[:,0])]
    return {'sound_speed_m_per_s': float(result.x[4]),
            'absorbers_xy_mm': pts.tolist()}
