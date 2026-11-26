import sys
sys.path.insert(0, '/tmp')
from test_proof_box import check
from math import comb
import numpy as np

# With identical columns any PSD minorant diagonal is zero. The best objective
# at cardinality k is attained by equal box-constrained coefficients.
p, y, l2, l0, M = 22, 10.4, .14, .22, 1.
k = np.arange(p + 1)
b = np.minimum(M, y / (k + l2))
f = .5*(y-k*b)**2 + k*(.5*l2*b*b+l0)
j = int(np.argmin(f))
# In a cube containing two optimal supports, an absent coordinate of either
# support is free. Adding 0.1 on it is permitted by the linear envelope.
a = .1
loss = (y-j)*a - .5*a*a - a*(l0+.5*l2)
print('leaf cap obstruction:', j, comb(p,j), f[j], np.partition(f,1)[1]-f[j], loss, flush=True)

# Opposing nearly identical columns can converge slowly when ridge is small.
# These remain small-support cases, so a failure is not explained by the cap.
for seed in range(5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(44,22)); X /= np.linalg.norm(X,axis=0)
    X[:,1] = X[:,0] + .003*X[:,1]
    y = 300*(X[:,0]-X[:,1]) + 1.5*X[:,3]
    try:
        print('small ridge', seed, check(dict(X=X,y=y,lam2=1e-7,lam0=.01,M=300.)), flush=True)
    except (AssertionError, RuntimeError) as e:
        print('small ridge failed', seed, type(e).__name__, str(e), flush=True)
