import pickle
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

with open('/holdout_task/public_thermal_cases.pkl', 'rb') as stream:
    cases = pickle.load(stream)
for i, case in enumerate(cases):
    m, gt = case['measurement'], case['gt']
    n, k = m['N'], m['kappa']
    ids = np.arange(n*n).reshape(n,n)
    a = np.r_[ids[:-1,:].ravel(), ids[:,:-1].ravel()]
    b = np.r_[ids[1:,:].ravel(), ids[:,1:].ravel()]
    ka, kb = k.ravel()[a], k.ravel()[b]
    c = 2*ka*kb/(ka+kb)
    A = coo_matrix((np.r_[c,c,-c,-c], (np.r_[a,b,a,b],np.r_[a,b,b,a])), shape=(n*n,n*n)).tocsc()
    wall = np.zeros((n,n))
    wall[[0,-1],:] += 1
    wall[:,[0,-1]] += 1
    d = wall.ravel()*m['gamma_robin']/n
    A = A + diags(d, format='csc')
    x = (np.arange(n)+.5)/n
    s,t = gt['source_xy']
    unit_load = np.exp(-.5*((x[:,None]-s)**2+(x[None,:]-t)**2)/m['sigma_m']**2)/n**2
    u = spsolve(A,unit_load.ravel())
    ix = ids[tuple(m['sensor_ij'].T)]
    pred = gt['amplitude']*u[ix]+gt['junction_offset']
    r = m['T_obs']-pred
    med = np.median(r)
    sig = 1.4826*np.median(abs(r-med))
    worst = np.argsort(abs(r-med))[-8:][::-1]
    print('case',i,'row-sum error',np.max(abs(A.sum(axis=1).A.ravel()-d)), 'balance error',abs(d@u-unit_load.sum()), 'unique sensors',len(np.unique(ix)))
    print('predicted range',np.ptp(pred),'residual median',med,'MAD sigma',sig,'largest eight residuals',r[worst])
    print('largest residual coordinates',m['sensor_ij'][worst].tolist())
