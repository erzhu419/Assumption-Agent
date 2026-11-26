from pathlib import Path

# Reuse the independent sparse comparison and certificate verifier, not its old cases.
source = Path('/tmp/verify_suc02_transfer.py').read_text()
exec(compile(source.split("\nout = check('public'", 1)[0], '/tmp/verify_suc02_transfer.py', 'exec'))

print('Public nuisance design shape:', public['Z_nuisance'].shape, 'ridge:', public['nuisance_ridge'])
raw, _ = mod._sensor_design(public)
z = public['Z_nuisance']
y = public['y_obs']
ridge = public['nuisance_ridge']
_, baseline = mod._ridge_residualize(raw, y, z, ridge)
assert np.allclose(baseline, gt['r_y'], rtol=0., atol=1e-11)
# A perturbation in the measured nuisance span is attenuated by the ridge
# residual operator, not annihilated as it would be by unpenalized centering.
a = np.linspace(-0.75, 1., z.shape[1])
_, shifted = mod._ridge_residualize(raw, y + z @ a, z, ridge)
u, s, _ = np.linalg.svd(z, full_matrices=False)
delta = z @ a
rejected = u.T @ delta
expected = delta - u @ rejected + u @ ((ridge / (s*s + ridge)) * rejected)
error = np.max(np.abs(shifted - baseline - expected))
print('Public nuisance-span perturbation max error:', error)
assert error < 1e-11

cap = float(public['u_box'])
n = raw.shape[1]
for side in (-1., 0., 1.):
    # Use the observed transport and outcome, but put budget slices at or
    # immediately beside integer-cap totals where an all-bound face can arise.
    budgets = cap * (np.array([1., n//2, n-1.]) + side * 1e-10)
    check(f'public measurements / cap breakpoint offset {side:+.0f}',
          dict(public, budget_grid=budgets), (gt['Ax'], gt['By']))
