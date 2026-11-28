import pickle
import numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']
x,y=c['true_channel_loads'],c['bridge_raw']; idx=np.arange(len(x)); axis=np.count_nonzero(x,axis=1)==1
A=np.c_[np.ones(len(x)),x]; w=np.ones(len(x))
for _ in range(12):
    p=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]
    e=np.linalg.norm(y-A@p,axis=1); w=1/np.sqrt(1+(e/np.median(e))**2)
clean=e<6*np.median(e)
for j in range(4):
    u,v=x[:,j],x[:,j^2]; z=y[:,j]
    for shared in (True,False):
        def design(freq,u,v):
            k,h=(freq[0],freq[-1]); return np.c_[np.ones(len(u)),u,np.sin(k*u),v,np.sin(h*v),np.cos(h*v)-1]
        def fit(tr,te):
            def profile(f):
                D=design(f,u[tr],v[tr]); p=np.linalg.lstsq(D,z[tr],rcond=None)[0]; return D,p
            def residual(f):
                D,p=profile(f); return D@p-z[tr]
            f=least_squares(residual,np.full(1 if shared else 2,2.4),bounds=(0.6,5.)).x
            _,p=profile(f); return design(f,u[te],v[te])@p, f,p
        err=np.zeros(len(x)); edge=[]
        for fold in range(5):
            te=clean&(idx%5==fold); err[te]=fit(clean&~te,te)[0]-z[te]
        for k in idx[axis&clean]:
            te=idx==k; edge.append((fit(clean&~te,te)[0]-z[te])[0])
        pred,f,p=fit(clean,clean)
        print(j,'shared',shared,'frequencies',f,'coef',p,'fit rms',np.std(pred-z[clean]),'mixed rms',np.std(err[clean&~axis]),'max fivefold/axis',max(abs(err)),max(abs(np.array(edge))))
        zz=np.linspace(-2.5,2.5,1001); print('derivative ranges',np.min(p[1]+f[0]*p[2]*np.cos(f[0]*zz)),np.max(abs(p[3]+f[-1]*(p[4]*np.cos(f[-1]*zz)-p[5]*np.sin(f[-1]*zz)))))
for part in ('train','test'):
    d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset'][part]
    for key in ('past_channel_raw','hold_channel_raw'):
        a=d[key].reshape(-1,4); print(part,key,a.min(0),a.max(0))
