import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('inversion', root / 'helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
exec(Path('/tmp/helmholtz_public_probe.py').read_text().split("if __name__ == '__main__':")[0])

rng = np.random.default_rng(7423)
worst = {}
for Lx, Ly, k, beta in [(1.0, 1.0, 11.5, .85), (1.2, .8, 12.2, .65), (.82, 1.16, 10.8, 1.05)]:
    p = dict(CASES[0]['problem'], Lx=Lx, Ly=Ly, k0=k, beta_imp=beta)
    shape = p['n_sq'].shape
    ref = reference_transfer(p)
    transfer_error = np.linalg.norm(solver._unit_source_fields(p).reshape(ref.shape) - ref) / np.linalg.norm(ref)
    assert transfer_error < 1e-12, (Lx, Ly, transfer_error)
    for mode in ('random', 'clustered'):
        errors = []
        for trial in range(16):
            a = rng.normal(size=ref.shape[1]); a *= 2.1 / np.linalg.norm(a)
            u = ref @ a
            phase = [np.pi, np.pi-0.0001, -np.pi+0.0001, 0.47][trial % 4]
            rotate = np.exp(1j * phase)
            known = rng.normal(size=12) + 1j*rng.normal(size=12)
            zp = rotate * known + .002 * (rng.normal(size=12) + 1j*rng.normal(size=12))
            idx = rng.choice(u.size, 140, replace=False)
            ij = np.array(np.unravel_index(idx, shape)).T
            z = rotate * u[idx] + .003 * (rng.normal(size=140) + 1j*rng.normal(size=140))
            if mode == 'random':
                bad = rng.choice(len(idx), 10, replace=False)
            else:
                center = ij[np.argmax(np.linalg.norm(ref[idx], axis=1))]
                bad = np.argsort(np.sum((ij-center)**2, axis=1))[:10]
            z[bad] += rng.uniform(.10,.75,size=10) * np.exp(1j*rng.uniform(-np.pi,np.pi,size=10))
            p['pilot_calibration'] = dict(known_field_re=known.real, known_field_im=known.imag, amplitude_obs=abs(zp), wrapped_phase_obs=np.angle(zp))
            p['survey'] = dict(sensor_ij=ij, amplitude_obs=abs(z), wrapped_phase_obs=np.angle(z), tikhonov_alpha=1e-6)
            out = solver.invert_helmholtz_sources_and_phase(p)
            t = out['global_phase_rad']
            assert isinstance(t,float) and -np.pi < t <= np.pi
            errors.append([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a), abs(np.angle(np.exp(1j*(t-phase)))), np.linalg.norm(out['predicted_field_abs'].ravel()-abs(u))/np.linalg.norm(u)])
        worst[(Lx, Ly, mode)] = np.max(errors, axis=0)
        print(f'{Lx:g} x {Ly:g}, {mode}: source/phase/field worst {worst[(Lx,Ly,mode)]}; transfer error {transfer_error:.3g}')
assert np.all(np.max(list(worst.values()), axis=0) <= [.030,.020,.030])
