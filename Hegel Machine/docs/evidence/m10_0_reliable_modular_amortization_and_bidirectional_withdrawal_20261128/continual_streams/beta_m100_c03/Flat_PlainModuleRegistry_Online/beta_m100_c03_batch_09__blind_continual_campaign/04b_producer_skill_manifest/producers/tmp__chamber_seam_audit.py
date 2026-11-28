import sys
from itertools import product
import numpy as np
sys.path[:0] = ['/holdout_task', '/tmp']
import wall_cv as cv
from downstream_solver import _responses
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

# Interpolate the two known endpoint stencils, one axis at a time.
s = cv.s
I = eye(cv.nx*cv.ny, format='csc')
axes = []
for d, n in zip('xy', (cv.nx, cv.ny)):
    args = s['kappa_'+d], s['vel_'+d], s['L'+d]/(n-1)
    a = cv.ar._axis(*args, False)
    axes.append((a, cv.ar._axis(*args, True)-a))
base = (kron(eye(cv.ny),axes[0][0])+kron(axes[1][0],eye(cv.nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
dx = kron(eye(cv.ny), axes[0][1]).tocsc()
dy = kron(axes[1][1], eye(cv.nx)).tocsc()
rows = np.arange(len(cv.y))
shift, sensor = divmod(rows, len(cv.indices))
rosette = sensor//4
families = list(product((0.,1.), repeat=2))
grid = list(product((0.,.25,.5,.75,1.,1.25),repeat=2))
def response(x,y):
    op = base+x*dx+y*dy
    return np.vstack([splu(op+mu*I).solve(cv.rhs)[cv.indices] for mu in s['mu_shifts']])
S = {p: response(*p) for p in grid}
actual, _ = _responses(s)
assert np.max(abs(actual-S[1.,1.])) < 1e-11
for src in (0., s['source_ridge']):
    for nz in (0., s['nuisance_ridge']):
        print('ridge(source,nuisance):',src,nz)
        for groups, label in ((rosette,'rosette'),(shift,'shift')):
            scores = {}
            for p in grid:
                folds = []
                for k in np.unique(groups):
                    test = groups==k
                    _, pred = cv.fit(S[p],~test,src,nz)
                    folds.append(np.sqrt(np.mean((pred[test]-cv.y[test])**2)))
                scores[p] = (float(np.sqrt(np.mean(np.square(folds)))),max(folds))
            print(label,'endpoint (pooled/worst)',[(p, np.round(scores[p],7).tolist()) for p in families], 'grid pooled minimum', min(scores,key=lambda p:scores[p][0]))
