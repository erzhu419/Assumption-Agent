import contextlib,io
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/chamber_probe.py').read().split('for name,s,u in')[0])
P,H=load(di)
X=m._features(di,di)[:len(H)]
D=th[2]+X@th[3:]
B=np.column_stack((1/taus,-np.expm1(-t[None,:]/taus[:,None])))
F=np.vstack((taus[:,None]*B,np.r_[1,t]))
Y=np.column_stack((di['instant_rate_obs'],y(di)))
E=Y-(H[:,None]*th[:2])@B

def fit(mask):
    # The normalized drift time response has unit derivative at zero.
    a=D[mask,None,None]*(F[:2]-F[2]).T[None,:,:]
    b=E[mask]-D[mask,None]*F[2]
    w=np.linalg.lstsq(a.reshape(-1,2),b.ravel(),rcond=None)[0]
    return np.r_[w,1-w.sum()]

labs=di['lab_id']
c=fit(np.ones(len(D),bool))
print('derivative-normalized drift fractions',c)
for j in np.unique(labs):
    use=labs==j
    coef=fit(~use)
    r=E[use]-D[use,None]*(coef@F)
    print('held-out lab',j,'coef',coef.round(5),'rate RMSE',np.sqrt(np.mean(r[:,0]**2)),'worst horizon RMSE',np.max(np.sqrt(np.mean(r[:,1:]**2,axis=0))))
for k in range(1,len(t)+1):
    cols=np.arange(F.shape[1])!=k
    a=D[:,None,None]*(F[:2,cols]-F[2,cols]).T[None,:,:]
    b=E[:,cols]-D[:,None]*F[2,cols]
    w=np.linalg.lstsq(a.reshape(-1,2),b.ravel(),rcond=None)[0]
    pred=D*(np.r_[w,1-w.sum()]@F[:,k]); print('held-out horizon',t[k-1],np.sqrt(np.mean((E[:,k]-pred)**2)))
np.save('/tmp/chamber_drift_fractions.npy',c)
