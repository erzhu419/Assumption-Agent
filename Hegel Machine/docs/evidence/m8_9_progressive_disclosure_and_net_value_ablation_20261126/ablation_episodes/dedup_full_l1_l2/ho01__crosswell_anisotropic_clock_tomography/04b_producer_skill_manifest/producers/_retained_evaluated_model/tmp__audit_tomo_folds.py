from pathlib import Path
import pickle
import numpy as np
from scipy.special import erf

path = Path('/holdout_task/tomography_solver.py')
source = path.read_text()
# Expose the fitted physical coefficients only in these in-memory audit copies.
needle = "    return {'plume_xz_m':"
assert source.count(needle) == 1

def load_variant(kind):
    text = source
    if kind == 'no_clocks':
        text = text.replace('clocks[j]))', 'clocks[j, :0]))')
        text = text.replace('clocks @ v[2:p]', 'np.zeros(len(rcv))')
    elif kind == 'extra_harmonic':
        text = text.replace('clocks[j]))', 'dz**4 / length**3, clocks[j]))')
        text = text.replace('clocks @ v[2:p]', 'clocks @ v[3:p]')
    text = text.replace(needle, "    return v[:p], v[p], {'plume_xz_m':")
    env = {}
    exec(compile(text, str(path), 'exec'), env)
    return env['invert_plume_and_clocks']

with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    data = pickle.load(f)
for kind in ('exact', 'no_clocks', 'extra_harmonic'):
    invert = load_variant(kind)
    metrics = []
    for row in data:
        s, gt = row['survey'], row['gt']
        obs = s['T_obs_s'] * 1000
        w, r = s['W_m'], s['R_plume_m']
        a, b = np.asarray(s['z_src_m'])[:, None], np.asarray(s['z_rcv_m'])[None, :]
        d = b - a
        L = np.hypot(w, d)
        i, j = np.indices(L.shape)
        for k in range(5):
            test = (i + 2*j) % 5 == k
            beta, amp, out = invert(dict(s, T_obs_s=np.where(test, np.nan, s['T_obs_s'])))
            x, z = out['plume_xz_m']
            clocks = np.asarray(out['rcv_clock_bias_s'])
            assert abs(clocks.sum()) < 1e-12 and max(abs(clocks)) <= .004 + 1e-14
            u = (w*x+d*(z-a))/L
            n = (-d*x+w*(z-a))/L
            I = np.sqrt(np.pi/2)*r*np.exp(-.5*(n/r)**2)*(erf(u/(np.sqrt(2)*r))+erf((L-u)/(np.sqrt(2)*r)))
            pred = L*beta[0] - d**2/L*beta[1] + amp*I + 1000*clocks
            if kind == 'extra_harmonic':
                pred += d**4/L**3*beta[2]
            e = (obs-pred)[test]
            # Retain a tail-sensitive metric as well as ordinary-pick error so an ablation cannot win by rejecting more picks.
            normal = abs(e) < 2.5
            metrics.append([np.sqrt(np.mean(e[normal]**2)), np.mean(np.minimum(abs(e), 2.5)), np.linalg.norm([x,z]-np.array(gt['plume_xz_m'])), 1000*np.sqrt(np.mean((clocks-gt['rcv_clock_bias_s'])**2)), np.sum(~normal)])
    m = np.asarray(metrics)
    print(kind, 'worst fold [ordinary RMS ms, clipped MAE ms, center m, clock RMSE ms, skip count]:', np.round(m.max(axis=0), 5), 'mean clipped MAE ms:', round(m[:, 1].mean(), 5), 'total skips:', int(m[:, -1].sum()))
