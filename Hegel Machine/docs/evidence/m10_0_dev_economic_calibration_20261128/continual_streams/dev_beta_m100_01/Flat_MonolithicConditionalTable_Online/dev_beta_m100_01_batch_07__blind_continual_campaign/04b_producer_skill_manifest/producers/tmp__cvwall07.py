from probe07 import m,s
import numpy as np
from itertools import product
y=m._calibrate(s); Z=s['Z_nuisance']; cap=s['u_box']; B=s['total_budget']; ids=s['sensor_ij']; nr=len(ids)//4; nmu=len(s['mu_shifts']); groups=np.tile(np.repeat(np.arange(nr),4),nmu); shift=np.repeat(np.arange(nmu),len(ids)); edge=np.tile((ids[:,0]==0)|(ids[:,1]==0)|(ids[:,0]==s['nx']-1)|(ids[:,1]==s['ny']-1),nmu)
def cv(S, folds, penalty):
    errors=[]
    for te in folds:
        tr=~te; A,b=m._project(dict(s,Z_nuisance=Z[tr],nuisance_ridge=penalty*s['nuisance_ridge']),S[tr],y[tr]); w,_=m._qp(A,b,penalty*s['source_ridge'],cap,B); z=Z[tr]; r=y[tr]-S[tr]@w
        g=np.linalg.solve(z.T@z+penalty*s['nuisance_ridge']*np.eye(z.shape[1]),z.T@r); e=y[te]-S[te]@w-Z[te]@g; errors.append(e)
    return np.round([np.sqrt(np.mean(np.concatenate(errors)**2)),max(np.sqrt(np.mean(e**2)) for e in errors)],7)
for walls in product(['reflect','periodic'],repeat=2):
    S,U=m._forward(s,walls)
    for p in [0,0.1,1]:
        print(walls,p,'rosette',cv(S,[groups==i for i in range(nr)],p),'shift',cv(S,[shift==i for i in range(nmu)],p),'wall/inside',cv(S,[edge,~edge],p))
