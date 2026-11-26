import sys
from pathlib import Path
exec(Path('/tmp/check_multicenter.py').read_text().split('for i, row in enumerate(')[0])
kind, seed = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(seed)
p = 22
q, _ = np.linalg.qr(rng.normal(size=(50, p)))
r, a, m = .14, .22, 1.1
if kind == 'public-like':
    g = np.eye(p)
    for k in (0, 11):
        for j in range(11):
            i, l = k+j, k+(j+1)%11
            g[i,l] = g[l,i] = rng.choice([-1,1])*.26
    b = rng.choice([-1,1],p) * rng.uniform(.54,.83,p)
elif kind == 'binding':
    m = .31
    u, _ = np.linalg.qr(rng.normal(size=(p,p)))
    g = (u * np.linspace(.35,1.65,p)) @ u.T
    b = rng.choice([-1,1],p) * rng.uniform(.77,1.13,p)
else:
    u, _ = np.linalg.qr(rng.normal(size=(p,p)))
    g = (u * np.geomspace(.06,2.0,p)) @ u.T
    b = rng.choice([-1,1],p) * np.sqrt(2*a*(g.diagonal()+r)) * rng.uniform(.82,1.13,p)
x = q @ np.linalg.cholesky(g).T
y = np.linalg.lstsq(x.T,b,rcond=None)[0]
data = dict(X=x, y=y, lam2=r, lam0=a, M=m)
start = time.monotonic()
result = mod._sparse(data)
print(kind, seed, 'leaves', len(result['subcube_leaves']), 'box hits', np.count_nonzero(abs(result['weights']) == m), 'seconds', round(time.monotonic()-start,3), flush=True)
gap, _ = check_cert(data,result)
print('verified gap',gap,flush=True)
assert 0 <= gap <= 1e-5
