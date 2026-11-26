import importlib.util
import numpy as np

path = '/holdout_task/alpha_w04_s00_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

truth = np.array([[1.04, -.05], [.97, .08], [1.11, -.09], [.91, .11]])
cmd = np.array([-2., -1.2, -.4, .3, .8, 1.5, 2.1])
def panel(commands, glitches=()):
    labs = np.repeat(np.arange(4), [len(x) for x in commands])
    xx = np.concatenate(commands)
    yy = truth[labs, 0]*xx + truth[labs, 1]
    for i, bump in glitches:
        yy[i] += bump
    return dict(lab_id=labs, cmd_stress=xx, true_stress=yy)

full = panel([cmd]*4)
scenarios = [
    ('EMI in both wings', panel([cmd]*4, [(0, .5), (9, -.3), (20, .8), (26, -.6)]), []),
    ('archive backs empty/noisy labs', panel([cmd, cmd[:0], cmd[:1], cmd], [(7, -.6)]), [full]),
    ('two separated clean checks per wing without archive', panel([cmd[::6], cmd, cmd[::6], cmd]), []),
]
for name, checks, archive in scenarios:
    m._archive_panels = lambda: iter(archive)
    err = m._calibrate(checks) - truth
    max_stress_err = np.max(np.abs(err[:, :1] * np.array([-2., 2.]) + err[:, 1:]))
    print(name, 'worst physical stress error:', round(max_stress_err, 8))
    assert max_stress_err < 1e-8, name

rng = np.random.default_rng(17)
hist = rng.uniform(-2, 2, (12, 18)); hold = rng.uniform(-2, 2, 12)
dt = rng.uniform(.1, .9, 18); t = np.r_[0., 1e-6, .25, 5.]; x = rng.normal(size=(12, 4))
for taus in (np.array([.2, 8.]), np.array([1.5, 1.51])):
    state = np.zeros((12, 2))
    for s, ds in enumerate(dt):
        f = np.exp(-ds / taus)
        state = state*f + (1-f)*hist[:, s, None]
    a, b = m._design(taus, dt, t, hist, hold, x)
    assert np.allclose(a[:, :, :2], (hold[:, None]-state)[:, None, :]*(1-np.exp(-t[:, None]/taus)))
    assert np.all(a[:, 0] == 0)
    assert np.allclose(a[:, 1]/t[1], b, atol=5e-5, rtol=5e-6)
print('Recurrence and onset derivative agree at the tau boundaries and for nearly coincident branches.')
