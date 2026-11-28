from probe07 import m,s
import numpy as np
from itertools import product
y=m._calibrate(s); Z=s['Z_nuisance']; ij=s['sensor_ij']; shifts=len(s['mu_shifts']); nr=len(ij)//4
ridge,cap,budget=(s[k] for k in ('source_ridge','u_box','total_budget'))
def evaluate(S, groups):
    err=np.empty(len(y)); norms=[]
    for te in groups:
        tr=~te; t=dict(s,Z_nuisance=Z[tr]); A,b=m._project(t,S[tr],y[tr]); w,_=m._qp(A,b,ridge,cap,budget)
        z=Z[tr]; gamma=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@(y[tr]-S[tr]@w)); err[te]=y[te]-S[te]@w-Z[te]@gamma
        norms.append(np.sqrt(np.mean(err[te]**2)))
    return np.sqrt(np.mean(err**2)),max(norms)
ros=np.tile(np.repeat(np.arange(nr),4),shifts); mu=np.repeat(np.arange(shifts),len(ij)); boundary=np.tile((ij[:,0]==0)|(ij[:,1]==0)|(ij[:,0]==s['nx']-1)|(ij[:,1]==s['ny']-1),shifts)
for walls in product(['open','reflect','periodic'],repeat=2):
    S,U=m._forward(s,walls); A,b=m._project(s,S,y); w,nu=m._qp(A,b,ridge,cap,budget); r=A@w-b
    print(walls,'full',round(np.sqrt(np.mean(r*r)),7),'rosette',np.round(evaluate(S,[ros==i for i in range(nr)]),7),'shift',np.round(evaluate(S,[mu==i for i in range(shifts)]),7),'boundary train/holdout',np.round(evaluate(S,[boundary,~boundary]),7),'w',w.round(4))
