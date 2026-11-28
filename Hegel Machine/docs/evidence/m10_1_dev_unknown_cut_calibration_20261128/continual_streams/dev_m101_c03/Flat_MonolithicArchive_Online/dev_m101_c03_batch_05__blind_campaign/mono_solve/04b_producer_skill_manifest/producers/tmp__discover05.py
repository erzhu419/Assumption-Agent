import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda D,Y:np.linalg.lstsq(D,Y,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
def terms(z):
    w=np.roll(z,-1,axis=-1)
    return np.stack((np.ones_like(z),z,w,z*w,z**3),axis=-1)
u,v=(s['panel_a'][k] for k in ['u','v']); d=terms(u);keep=np.ones(len(u),bool)
for _ in range(6):
    c=np.array([fit(d[keep,j],v[keep,j]) for j in range(4)]);r=np.linalg.norm((d*c).sum(-1)-v,axis=-1);keep=r<6*np.median(r)
print('A archive cyc',c,'inliers',keep.sum(),'largest clean errors',np.sort(r)[-8:])
for name in ['v1','v2']:
    u=s['panel_c']['u'];v=s['panel_c'][name]
    def loss(D):
        R=D@fit(D,v)-v; h=(np.linalg.svd(D,full_matrices=False)[0]**2).sum(1);return np.sqrt(np.mean(R**2)),np.sqrt(np.mean((R/(1-h[:,None]))**2))
    p=[(i,j) for i,j in combinations_with_replacement(range(4),2)]; cross=np.array([u[:,i]*u[:,j] for i,j in p if i!=j]).T
    for label,D in [('lin',aff(u)),('cross',aff(cross)),('square',aff(u*u)),('quad',aff(np.c_[u,u*u,cross])),('cubic',aff(np.c_[u,u**3]))]: print(name,label,loss(D))
    for label,f in [('cos',lambda k:aff(np.cos(k*u))),('sin',lambda k:aff(np.sin(k*u))),('sinlin',lambda k:aff(np.c_[u,np.sin(k*u)])),('tanh',lambda k:aff(np.tanh(k*u))),('tanhl',lambda k:aff(np.c_[u,np.tanh(k*u)])),('rational',lambda k:aff(u/(1+k*u*u)))]:
        o=minimize_scalar(lambda k:loss(f(k))[0],bounds=(.15,5),method='bounded');print(name,label,o.x,loss(f(o.x)))
    o=least_squares(lambda k:(aff(u+k[0]*u*u)@fit(aff(u+k[0]*u*u),v)-v).ravel(),[0.]);print(name,'x+kx2',o.x,loss(aff(u+o.x[0]*u*u)))
t=s['eval_times'];Y=s['train']['traj'];y=s['train']['y']
def temporal(k):return -np.expm1(-t[:,None]*k)
for q in [1,2]:
    o=least_squares(lambda k:(temporal(k)@fit(temporal(k),Y.T)-Y.T).ravel(),[1.] if q==1 else [.45,1.9],bounds=(.001,12)); print('time',o.x,'rmse',np.sqrt(np.mean(o.fun**2)));print('corr amplitudes, y',np.corrcoef(np.c_[fit(temporal(o.x),Y.T).T,y].T))
