from pathlib import Path
import importlib.util
import pickle
import numpy as np

root = Path('/holdout_task')
def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = load(root / 'beta_m97_c01_batch_01_solver.py')
prior = load(Path('/tmp/hegel_continual_archive/beta_m97_c01_s00_solver.py'))
# Use the published verifier without running its command-line entry point.
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
solve = mod.solve_beta_m97_c01_batch_01
items = pickle.loads((root / 'public_beta_m97_c01_batch_01.pkl').read_bytes())
for item in items:
    s, gt = item['study'], item['gt']
    out = solve(s)
    cert = out['kkt_certificate']
    np.testing.assert_array_equal(out['weights'], cert['weights'])
    np.testing.assert_allclose(out['weights'], gt['w_opt'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(out['terminal_field'], gt['terminal_field_A'], atol=1e-7, rtol=0)
    v = verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], s['total_budget'], cert)
    assert v['valid'], v
    print('public independent certificate gap', v['gap'])
    aux = dict(s, aux_manifold_telemetry=np.full_like(s['aux_manifold_telemetry'], 123.))
    np.testing.assert_array_equal(solve(aux)['weights'], out['weights'])
    sealed_other_robin = dict(s, robin_x=19., robin_y=23.)
    np.testing.assert_array_equal(mod._forward(sealed_other_robin)[0], mod._forward(s)[0])
    for case in (dict(s, past_vortex_circulation_trace=[s['critical_vortex_circulation']]), dict(s, recirculation_loop_mode=1)):
        S, F = mod._forward(case)
        Sp, Fp = prior._forward(case)
        np.testing.assert_array_equal(S, Sp)
        np.testing.assert_array_equal(F, Fp)

s = items[0]['study']
sweep = {k: np.array(v, copy=True) for k,v in s['sensor_calibration_sweeps'].items()}
banks = sweep['bank_id']
original = mod._archived_affine
calls = []
def record(bank):
    calls.append(int(bank))
    return original(bank)
mod._archived_affine = record
base = mod._calibrated_observations(s)
for bank in range(4):
    ix = np.flatnonzero(banks == bank)
    rows = np.asarray(s['row_bank_id']) == bank
    g,o = original(bank)
    for retained, spike, fallback in [(1,0,True),(2,0,False),(4,0,False),(4,3.2,True),(len(ix),-4.,False)]:
        keep = (banks != bank) | np.isin(np.arange(len(banks)), ix[:retained])
        test = {k:v[keep].copy() for k,v in sweep.items()}
        j = np.flatnonzero(test['bank_id'] == bank)[-1]
        test['raw_readout'][j] += spike
        calls.clear()
        arr = mod._calibrated_observations(dict(s, sensor_calibration_sweeps=test))
        assert calls == ([bank] if fallback else []), (bank,retained,spike,calls)
        np.testing.assert_array_equal(arr[~rows], base[~rows])
        expected = (np.asarray(s['y_obs_raw'])[rows]-o)/g if fallback else base[rows]
        np.testing.assert_allclose(arr[rows], expected, atol=2e-5, rtol=0)
mod._archived_affine = original
print('calibration: 20 sparse/clean/spike slices passed across four banks')

rng = np.random.default_rng(7401)
worst = 0.
count = 0
for i in range(80):
    S = rng.normal(size=(30,12)) if i % 2 else rng.normal(size=(30,3)) @ rng.normal(size=(3,12))
    y = 10*rng.normal(size=30)
    ridge = (1e-5,0.03,2.)[i%3]
    cap = (0.17,1.,2.2)[i%3]
    for b in [0.,cap*1e-6,cap,5*cap,5.35*cap,cap*(12-1e-6),12*cap]:
        w,nu = mod._box_simplex(S,y,ridge,cap,b)
        r = S@w-y
        g = S.T@r+ridge*w
        lo,hi = np.maximum(g-nu,0),np.maximum(nu-g,0)
        p = .5*(r@r+ridge*(w@w))
        cert = dict(weights=w,nu_sum=nu,lam_lower=lo,lam_upper=hi,primal_objective=p,dual_lower_bound=p-lo@w-hi@(cap-w))
        result = verify(S,y,ridge,cap,b,cert)
        assert result['valid'], (i,b,result)
        worst = max(worst,result['gap'])
        count += 1
print('QP stress certificate count', count, 'worst verified gap', worst)
