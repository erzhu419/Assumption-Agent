from pathlib import Path
exec(Path('/tmp/verify_eq02.py').read_text().split('for seed, kind in enumerate')[0])

public = root / 'public_eq02_studies.pkl'
validator = root / 'validate_public_schema.py'
if public.is_file() and validator.is_file():
    import subprocess, sys
    subprocess.run([sys.executable, '-I', str(validator)], cwd=root, check=True)
else:
    print('Public fixtures unavailable; the following are synthetic tolerance tests.', flush=True)

# Weakly correlated predictors with all marginal scores near the activation
# boundary can require a larger partition than sparse-signal examples.
worst = []
for i in range(12):
    rng = np.random.default_rng(20260+i)
    p, n = 18 + i%2, 96
    common = rng.normal(size=(n,1))
    x = rng.normal(size=(n,p)) + (0.18 + 0.15*(i%4))*common*rng.choice([-1,1],p)
    x /= np.linalg.norm(x, axis=0)
    l0, l2, box = .125, .015, (.45 if i%3 == 0 else 1.)
    b = .50 * rng.choice([-1,1],p) * rng.uniform(.94,1.06,p)
    y = x @ np.linalg.solve(x.T @ x,b)
    t = perf_counter()
    try:
        cert = mod._BoxSearch(x,y,l0,l2,box).solve()
        dt = perf_counter()-t
        gap, leaves, slack = replay(x,y,l0,l2,box,cert)
        worst.append((dt,i,gap,leaves))
        print(f'case={i} p={p} M={box}: gap={gap:.4g} leaves={leaves} seconds={dt:.3f}', flush=True)
    except RuntimeError as e:
        print(f'case={i} p={p} M={box}: FAIL after {perf_counter()-t:.3f}s: {e}', flush=True)
print('two-slowest combined seconds:', sum(z[0] for z in sorted(worst,reverse=True)[:2]), flush=True)
