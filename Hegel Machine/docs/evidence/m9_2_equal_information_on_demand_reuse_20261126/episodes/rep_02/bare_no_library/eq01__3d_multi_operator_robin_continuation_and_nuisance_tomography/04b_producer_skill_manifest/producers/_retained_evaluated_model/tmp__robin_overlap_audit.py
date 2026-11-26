import runpy
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve

ns = runpy.run_path('/tmp/robin_stress.py')
s = ns['study']((8, 7, 6), 1)
sensors = np.ravel_multi_index(s['sensor_ijk'].T, s['kappa_base'].shape)
reference = []
for theta in s['thetas']:
    a, vol = ns['independent_matrix'](s, theta)
    rhs = vol * s['source_modes'].reshape(3, -1).T
    reference.append(spsolve(a, rhs)[sensors])
S = np.concatenate(reference)
rng = np.random.default_rng(93)
unit = S / np.linalg.norm(S, axis=0)
Z = rng.normal(size=(len(S), 7))
Z[:, :3] = unit + 1e-6 * rng.normal(size=unit.shape)
Z[:, 6] = Z[:, 0] + Z[:, 1]
s['Z_nuisance'] = Z
s['nuisance_ridge'] = 1e-8
s['Y_obs'] = S @ rng.normal(size=(3, 5)) + Z @ rng.normal(size=(7, 5)) + 1e-4 * rng.normal(size=(len(S), 5))
out = ns['mod'].solve_robin_continuation_tomography(s)
profile, joint = ns['check_inverse'](s, out, S)
print(f'Nearly coincident source/nuisance designs: profile error {profile:.3e}, joint error {joint:.3e}')
assert profile <= 1e-5 and joint <= .03
if not Path('/holdout_task/public_eq01_studies.pkl').is_file():
    print('Public measurements are not mounted; overlap audit used a synthetic study.')
