import importlib.util
import itertools
import pickle

import numpy as np
from scipy.optimize import lsq_linear

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'certified_l0_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def block_opt(X, y, ridge, box, penalty):
    p = X.shape[1]
    A = np.vstack((X, np.sqrt(ridge)*np.eye(p)))
    b = np.r_[y, np.zeros(p)]
    best = np.inf
    for bits in itertools.product((False, True), repeat=p):
        radii = np.where(bits, box, 1e-9)
        D = A*radii
        fit = lsq_linear(D,b,bounds=(-1.,1.),method='bvls',tol=1e-14)
        e = D@fit.x-b
        best = min(best, .5*(e@e)+penalty*sum(bits))
    return best

for i, a in enumerate(pickle.load(open(root+'public_box_l0_instances.pkl','rb'))):
    source = a['X']
    blocks = np.array_split(np.arange(source.shape[0]), 7)
    for perturb in (0., .003):
        X = np.zeros((len(source), 15))
        y = np.zeros(len(source))
        ridge, box, penalty = .012, .6, .15
        ref = 0.
        j = 0
        for k, rows in enumerate(blocks):
            width = 3 if k == 0 else 2
            v = source[rows,k]; v = v/np.linalg.norm(v)
            Z = source[np.ix_(rows, np.arange(k+1,k+1+width))]
            Z = Z / np.linalg.norm(Z,axis=0)
            D = (v[:,None]+perturb*Z)*np.where(np.arange(width)%2,-1.,1.)
            target = (.68 if k%2 else 1.04)*v
            X[np.ix_(rows,np.arange(j,j+width))] = D
            y[rows] = target
            ref += block_opt(D,target,ridge,box,penalty)
            j += width
        r = m.solve_certified_box_l0(dict(X=X,y=y,ridge=ridge,box=box,l0_penalty=penalty))
        w = np.asarray(r['weights']); e = X@w-y
        obj = .5*(e@e+ridge*(w@w))+penalty*np.count_nonzero(np.abs(w)>1e-9)
        lb, nodes = r['certified_lower_bound'], r['nodes_explored']
        assert np.max(np.abs(w))<=box and abs(obj-r['objective'])<1e-12 and 0<=nodes<=600
        print(f'public-{i}, perturb={perturb}: lower-reference={lb-ref:.8g}, objective-reference={obj-ref:.8g}, gap={obj-lb:.8g}, nodes={nodes}',flush=True)
        assert lb<=ref+1e-11 and ref<=obj+1e-11
