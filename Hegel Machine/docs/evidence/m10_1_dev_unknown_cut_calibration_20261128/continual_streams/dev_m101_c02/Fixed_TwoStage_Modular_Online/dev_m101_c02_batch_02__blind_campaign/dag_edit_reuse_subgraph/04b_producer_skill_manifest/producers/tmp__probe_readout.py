import itertools as it
import pickle
import numpy as np
from scipy.optimize import minimize_scalar
from pathlib import Path
ns={}; exec(Path('_frozen_nodes/node_00.py').read_text(),ns)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; z=ns['run_node']({'__input__':s})['z_tr']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; c=s['panel_c']; n=len(z)
def poly(x,d):
    idx=[j for k in range(d+1) for j in it.combinations_with_replacement(range(x.shape[1]),k)]
    return np.stack([np.prod(x[:,j],axis=1) for j in idx],axis=1),idx
def cv(M,Y,folds):
    e=[]
    for valid in folds:
        fit=np.setdiff1d(np.arange(len(Y)),valid); e.append(np.mean(np.abs(M[valid]@np.linalg.lstsq(M[fit],Y[fit],rcond=None)[0]-Y[valid])))
    return np.round(e,5)
folds=[np.arange(52,72),*np.array_split(np.arange(n),6)]
print('coordinates ranges',z.min(0),z.max(0)); print('scalar/traj std',y.std(),tr.std(0)); print('traj singular values',np.linalg.svd(tr,compute_uv=False)); print('first row traj',tr[0]); print('corr y and terminal traj',np.corrcoef(y,tr[:,-1])[0,1]); print('cv order official then six contiguous folds')
for d in range(1,5):
    M,idx=poly(z,d); co=np.linalg.lstsq(M,y,rcond=None)[0]; print('degree',d,'scalar',cv(M,y,folds),'traj',cv(M,tr,folds),'top scalar',[(idx[i],round(co[i],4)) for i in np.argsort(abs(co))[-12:][::-1]])
for offset in (False,True):
    def err(k):
        g=-np.expm1(-k*t); D=np.stack((np.ones_like(g),g),axis=1) if offset else g[:,None]; return np.mean((tr.T-D@np.linalg.lstsq(D,tr.T,rcond=None)[0])**2)
    k=minimize_scalar(err,bounds=(0.01,4),method='bounded').x; print('common decay',offset,k,err(k)**.5)
for key in ('v1','v2'):
    for d in (1,2):
        M,idx=poly(c['u'],d); V=c[key]; co=np.linalg.lstsq(M,V,rcond=None)[0]; print('panel',key,'degree',d,'cv',cv(M,V,np.array_split(np.arange(len(V)),4)),'coeffs top', [[(idx[i],round(col[i],4)) for i in np.argsort(abs(col))[-4:][::-1]] for col in co.T])
