import runpy
import numpy as np
from scipy.optimize import minimize_scalar
v = runpy.run_path('/tmp/verify02.py')
m,s,c,z,y,splits = (v[k] for k in ('m','s','c','z','y','splits'))
p = s['panel_a']; u=p['u']; obs=p['v']; mats=[m._calibration_features(u,j) for j in range(4)]
res=obs-np.column_stack([a@w for a,w in zip(mats,c)])
rows=np.flatnonzero(np.linalg.norm(res,axis=1)<0.01)
# Compare channel-specific and shared nonlinear coefficients on held-out panel rows.
A=np.zeros((len(u),4,11))
for j,h in enumerate(mats):
    A[:,j,2*j:2*j+2]=h[:,:2]
    A[:,j,8:]=h[:,2:]
err=[]
for h in rows:
    f=rows[rows!=h]
    local=np.array([a[h]@m._ls(a[f],obs[f,j]) for j,a in enumerate(mats)])
    shared=A[h]@m._ls(A[f].reshape(-1,11),obs[f].ravel())
    err.append(np.abs(np.array([local,shared])-obs[h]))
err=np.array(err); print('panel jackknife local/shared MAE and maximum',err.mean((0,2)),err.max((0,2)))
# Test whether bounded rational curvature helps beyond the paired interaction.
# Denominator choice and strength use only the outer training rows.
S=m._scalar(z); q=[z[:,j] for j in range(4)]+[z[:,j]+sg*z[:,k] for j in range(4) for k in range(j+1,4) for sg in (-1,1)]+[S[:,-1]]
D=[np.ones((len(z),1))]+[1+a*(w[:,None]/scale)**2 for w,scale in [(w,max(abs(w))) for w in q] for a in (0.001,0.005,0.02,0.1,0.5)]
errs=[]
for i in splits:
    f,h=i[:56],i[56:]; scores=[]
    for d in D:
        X=S/d; train=X[f]; Q,_=np.linalg.qr(train,mode='reduced'); e=y[f]-Q@(Q.T@y[f]); scores.append(np.mean(abs(e/(1-(Q*Q).sum(1)))))
    j=np.argmin(scores); X=S/D[j]; baseline=S[h]@m._ls(S[f],y[f]); alt=X[h]@m._ls(X[f],y[f]); errs.append((np.mean(abs(baseline-y[h])),np.mean(abs(alt-y[h])))); print('rational audit',len(errs)-1,'selected',j,'MAE',errs[-1])
print('rational audit mean and worst',np.mean(errs,axis=0),np.max(errs,axis=0))
