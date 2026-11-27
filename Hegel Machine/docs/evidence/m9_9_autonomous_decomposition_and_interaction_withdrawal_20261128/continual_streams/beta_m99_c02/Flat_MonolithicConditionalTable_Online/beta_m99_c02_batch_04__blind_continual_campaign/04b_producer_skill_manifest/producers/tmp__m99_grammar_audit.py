import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; n=len(x)

def cyclic(x,j):
    a,b,d=x[:,j],x[:,(j+1)%4],x[:,(j-1)%4]
    return np.c_[np.ones(len(x)),a,b,d,a**3,a*b]

def design(x,j,model):
    d=cyclic(x,j)
    if model=='full': return m._features(x,j)
    if model=='expanded':
        a,b,p=x[:,j],x[:,(j+1)%4],x[:,(j-1)%4]
        return np.c_[m._features(x,j),x*x,a*b*p,d[:,1:]*(a*a)[:,None],d[:,1:]*((a-b)**2)[:,None],d[:,1:]*((a+b)**2)[:,None]]
    return d

def robust(d,v):
    b=np.linalg.lstsq(d,v,rcond=None)[0]
    for _ in range(55):
        w=1/np.sqrt(1+((v-d@b)/.006)**2)
        b=np.linalg.lstsq(d*w[:,None],v*w,rcond=None)[0]
    good=abs(v-d@b)<.015
    return np.linalg.lstsq(d[good],v[good],rcond=None)[0],good

# Nested within each fold: select corruptions from training residuals only.
for model in ('full','cyclic','tied','expanded'):
    e=np.zeros_like(y)
    for k in range(5):
        test=np.arange(n)%5==k; train=~test
        d=np.stack([design(x,j,model) for j in range(4)],axis=1)
        d0=np.stack([cyclic(x,j) for j in range(4)],axis=1)
        fits=[robust(d0[train,j],y[train,j]) for j in range(4)]
        good=np.stack([f[1] for f in fits],axis=1)
        if model=='tied':
            b=np.stack([f[0] for f in fits]); start=np.r_[b[:,0],b[:,1],np.mean(b[:,2:]/b[:,1,None],axis=0)]
            def predict(z,p):
                return p[:4]+p[4:8]*(z[:,:,1]+z[:,:,2:]@p[8:])
            p=least_squares(lambda p:(predict(d[train],p)-y[train])[good],start,gtol=1e-13,ftol=1e-13,xtol=1e-13).x
            e[test]=predict(d[test],p)-y[test]
        else:
            for j in range(4):
                keep=np.flatnonzero(train)[good[:,j]]
                b=np.linalg.lstsq(d[keep,j],y[keep,j],rcond=None)[0]
                e[test,j]=d[test,j]@b-y[test,j]
    print(model, 'gross sweep cells',np.argwhere(abs(e)>1).tolist(), 'clean max-channel RMSE',max(np.sqrt(np.mean(e[abs(e[:,j])<1,j]**2)) for j in range(4)), 'clean max abs',np.max(abs(e[abs(e)<1])))
S,_=m._responses(s); y=m._calibrate(s); z=s['Z_nuisance']; bank=s['row_bank_id']; shift=np.repeat(np.arange(len(s['mu_shifts'])),len(s['sensor_ij']))
for mask_set in [bank,shift]:
    for k in np.unique(mask_set):
        hold=mask_set==k; fit=~hold; p={**s,'Z_nuisance':z[fit]}; sm=S[fit]; yy=y[fit]
        X,r=m._profile(sm,yy,p); ridge=s['source_ridge']; args=(s['u_box'],s['total_budget'])
        w,_=m._box_simplex(X.T@X+ridge*np.eye(X.shape[1]),-X.T@r,*args)
        wu,_=m._box_simplex(sm.T@sm+ridge*np.eye(X.shape[1]),-sm.T@yy,*args)
        gamma=np.linalg.solve(z[fit].T@z[fit]+s['nuisance_ridge']*np.eye(z.shape[1]),z[fit].T@(yy-sm@w))
        print('group',k,'held-out RMSE unadjusted/profiled',*[np.sqrt(np.mean(e*e)) for e in (S[hold]@wu-y[hold],S[hold]@w+z[hold]@gamma-y[hold])])
