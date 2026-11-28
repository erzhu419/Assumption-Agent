import itertools
import pickle
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

def rmse(x):
    return np.sqrt(np.mean(x*x))

def alternate_axis(k, v, h, robin, seam, adv, leak):
    a = dn._axis(k, v, h).tolil()
    for i, j, p in [(0, len(k)-1, 1), (len(k)-1, 0, len(k)-2)]:
        side = 1 if i == 0 else -1
        inner = (k[i]+k[p])/(2*h*h) + adv*max(-side*v[i],0)/h
        coeff = {'mid': (k[i]+k[j])/2, 'harmonic': 2*k[i]*k[j]/(k[i]+k[j]), 'local': k[i], 'reflect': 0}[seam]
        outer = coeff/(h*h) + (adv and seam != 'reflect')*max(side*v[i],0)/h
        a[i,p], a[i,j], a[i,i] = -inner, -outer, inner+outer+leak*robin/h
    return a.tocsc()

for item in items:
    s = item['study']
    pulses = item['campaign_diagnostics']
    W = np.stack([p['w_pulse'] for p in pulses])
    def measure(payload):
        return np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),payload)['y_obs_raw'] for p in pulses])
    y = measure(batch._PAY)
    y_fit = measure(None)
    cal = calibrate_upstream(s,batch._PAY)
    production = W @ dn._forward(cal)[0].T
    nx,ny = s['nx'],s['ny']
    i,j = s['sensor_ij'].T
    bank = np.asarray(s['row_bank_id'])
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ W.T
    eye = sparse.eye(nx*ny,format='csc')
    reaction = sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    opts = [('mid',1,0),('mid',0,0),('harmonic',1,0),('local',1,0),('mid',1,1),('reflect',0,0),('reflect',0,1)]
    ops = [[alternate_axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(n-1),s['robin_'+a],*o) for o in opts] for a,n in [('x',nx),('y',ny)]]
    errors = []
    for ix,iy in itertools.product(range(len(opts)),repeat=2):
        L = sparse.kron(sparse.eye(ny),ops[0][ix],format='csc') + sparse.kron(ops[1][iy],sparse.eye(nx),format='csc') + reaction
        pred = np.vstack([splu(L+mu*eye).solve(rhs)[i+nx*j] for mu in s['mu_shifts']]).T
        if (ix,iy)==(0,0):
            np.testing.assert_allclose(pred,production,atol=1e-12,rtol=0)
        errors.append(pred-y)
    E = np.asarray(errors)
    max_held = 0.
    for p in range(len(pulses)):
        for b in np.unique(bank):
            train = E[:,np.arange(len(pulses))!=p][:,:,bank!=b]
            mse = (train**2).mean(axis=(1,2))
            rank = np.argsort(mse)
            held = rmse(E[rank[0],p,bank==b])
            print('held pulse/bank',p,int(b),'winner',int(rank[0]),'held RMSE',held,'train runner-up/winner',np.sqrt(mse[rank[1]]/mse[rank[0]]))
            assert rank[0]==0
            max_held = max(max_held,held)
    print('worst held pulse/bank RMSE',max_held,'worst pulse RMSE staged',max(map(rmse,production-y)),'worst pulse RMSE sweep refit',max(map(rmse,production-y_fit)))
    assert max_held < 1e-5
