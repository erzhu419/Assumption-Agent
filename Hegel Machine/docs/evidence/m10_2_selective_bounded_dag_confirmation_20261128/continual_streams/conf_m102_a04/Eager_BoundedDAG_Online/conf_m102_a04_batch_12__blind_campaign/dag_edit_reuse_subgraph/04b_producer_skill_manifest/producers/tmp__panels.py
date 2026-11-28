import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=150)
def lin(X,y): return np.linalg.lstsq(X,y,rcond=1e-10)[0]
def irls(X,y):
    c=lin(X,y)
    for _ in range(18):
        e=y-X@c; s=np.maximum(np.median(abs(e),axis=0),.0002); w=1/np.sqrt(1+(e/(3*s))**2)
        c=np.stack([lin(X*w[:,i,None],y[:,i]*w[:,i]) for i in range(y.shape[1])],1)
    return c
for k in ['panel_a','panel_b','panel_c']:
    u=b[k]['u']; print('\n',k,'u range',u.min(0),u.max(0)); one=np.ones((len(u),1)); U=np.c_[one,u]
    for v in [a for a in b[k] if a!='u']:
        y=b[k][v]; print(v,'range',y.min(0),y.max(0),'affine_coef',irls(U,y)); print('affine residual quantiles',np.quantile(abs(y-U@irls(U,y)),[.1,.5,.8,.9,1],axis=0))
        fs={'sin':(lambda q:np.c_[U,np.sin(q[0]*u)], [.9],([.05],[5.])), 'sin_cos':(lambda q:np.c_[U,np.sin(q[0]*u),np.cos(q[0]*u)], [.9],([.05],[5.])), 'quad':(lambda q:np.c_[U,*[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]],[1],([0],[2])), 'softsign':(lambda q:np.c_[one,u/(1+q[0]*abs(u))+q[1]*u],[1,0],([.02,-1],[10,1])), 'curvature':(lambda q:np.c_[U,u*u,u*u/(1+q[0]*u*u)],[.25],([.02],[4])), 'tanh':(lambda q:np.c_[U,np.tanh(q[0]*u)],[1],([.02],[6])), 'mix_sin':(lambda q:np.c_[U,np.sin(q[0]*u)],[2.15],([.05],[5.]))}
        for name,(fn,q,bounds) in fs.items():
            def res(q):
                X=fn(q); return (y-X@(irls(X,y) if k=='panel_a' else lin(X,y))).ravel()
            fit=least_squares(res,q,bounds=bounds,loss='soft_l1' if k=='panel_a' else 'linear',f_scale=.004); e=abs(fit.fun); print(name,fit.x,'med/rms/max',np.median(e),np.sqrt(np.mean(e**2)),max(e))
        print('first u / y',np.c_[u,y][:4])
