import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; U,V=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(V),axis=1)<3; u,v=U[keep],V[keep]
D=u.shape[1]; ind=(np.arange(D)+2)%D

def fit(p,u,v,full=True):
    m=[]; r=[]
    for j in range(D):
        i=ind[j]; a=np.c_[np.ones(len(u)),u if full else u[:,[j]],np.tanh(p[0]*u[:,j]),np.sin(p[1]*u[:,i])]; c=np.linalg.lstsq(a,v[:,j],rcond=None)[0]; m.append(c); r.append(a@c-v[:,j])
    return np.array(r).T,np.array(m).T
for full in (False,True):
 p=least_squares(lambda p: fit(p,u,v,full)[0].ravel(),[1.8,2.4]).x; e,m=fit(p,u,v,full); print('parameters',p,'RMS',np.sqrt(np.mean(e**2,axis=0)),'coefficients\n',m); print('fixed frequency RMS',np.sqrt(np.mean(fit([1.8,2.4],u,v,full)[0]**2,axis=0)))
 er=[]
 for k in range(len(u)):
    a=np.arange(len(u))!=k; p=least_squares(lambda p: fit(p,u[a],v[a],full)[0].ravel(),[1.8,2.4]).x; _,m=fit(p,u[a],v[a],full); z=u[k]; er.append(m[0]+(z@m[1:1+D] if full else z*m[1])+np.tanh(p[0]*z)*m[-2]+np.sin(p[1]*z[ind])*m[-1]-v[k])
 print('LOO RMSE',np.sqrt(np.mean(np.array(er)**2,axis=0)))
print('A shapes/rejected',U.shape,np.where(~keep)[0]); print('sample ranges',np.min(u,axis=0),np.max(u,axis=0)); print('study keys',s.keys())
