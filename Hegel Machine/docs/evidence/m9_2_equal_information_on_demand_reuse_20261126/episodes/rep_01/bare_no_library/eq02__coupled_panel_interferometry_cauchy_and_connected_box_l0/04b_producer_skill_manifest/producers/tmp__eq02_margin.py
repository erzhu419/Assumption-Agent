import runpy, time, warnings
import numpy as np
v = runpy.run_path('/tmp/eq02_verify.py')
m, verify = v['mod'], v['verify']
# Connected, nearly independent pairs with competing supports close to the L0 threshold.
for seed in range(318, 326):
    rng = np.random.default_rng(seed)
    p = 18 + seed % 2
    base = rng.normal(size=(95, (p+1)//2))
    x = base[:, np.arange(p)//2] + (0.23 + 0.08*(seed%4))*rng.normal(size=(95,p))
    x *= np.exp(rng.uniform(-0.6, 0.6, p)) / np.linalg.norm(x, axis=0)
    y = x @ (rng.choice([-1., 1.], p)*rng.uniform(.22,.67,p)) + .09*rng.normal(size=len(x))
    l2, l0, M = .025, .10 + .015*(seed%3), .55 + .20*(seed%2)
    t = time.monotonic(); cert = m._box_l0(x,y,l2,l0,M); dt = time.monotonic()-t
    _, gap, n = verify(x,y,l2,l0,M,cert)
    print(f'paired seed={seed} p={p} gap={gap:.3g} leaves={n} seconds={dt:.3f}', flush=True)
    assert -1e-10 <= gap <= 1e-5 and n <= 2048 and dt <= 9
# With zero sparsity penalty the problem is ordinary box ridge. Warnings here
# expose a singular envelope formula even if the search accidentally succeeds.
with warnings.catch_warnings():
    warnings.simplefilter('error', RuntimeWarning)
    cert = m._box_l0(x, y, l2, 0., M)
    _, gap, n = verify(x,y,l2,0.,M,cert)
    print(f'zero-penalty limit: gap={gap:.3g} leaves={n}', flush=True)
    assert -1e-10 <= gap <= 1e-5 and n <= 2048
