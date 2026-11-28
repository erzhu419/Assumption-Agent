import numpy as np, pickle, glob, importlib.util
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def load(i):
    p=glob.glob(f'/tmp/hegel_continual_archive/*{i}_solver.py')[0]; spec=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('06'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,b,z,v=m._sweeps(s)
X=np.vstack((x,np.eye(4)[b]*z[:,None])); Y=np.vstack((y,np.where(np.eye(4)[b]>0,v[:,None],np.nan)))

def fit(A,Y):
    init=np.linalg.lstsq(A,Y,rcond=None)[0]
    p=least_squares(lambda p:A@p-Y,init,loss='soft_l1',f_scale=.003).x
    return p

def check(name, design):
    out=[]
    for j in range(4):
        a=design(X,j); t=Y[:,j]; fold=np.arange(len(t))%5; e=np.full(len(t),np.nan)
        for k in range(5):
            keep=np.isfinite(t)&(fold!=k); p=fit(a[keep],t[keep]); valid=np.isfinite(t)&(fold==k); e[valid]=a[valid]@p-t[valid]
        keep=np.isfinite(t); p=fit(a[keep],t[keep]); print(name,j,np.quantile(abs(e[keep]),[.5,.8,.95]).round(6),'fit',p.round(5)); out.append(p)
    return out

if __name__=='__main__':
 for name,design in [('linear',lambda x,j:np.c_[np.ones(len(x)),x]),('old04',load('04')._design),('degree2',lambda x,j:np.array([np.prod(x[:,k],axis=1) for d in range(3) for k in combinations_with_replacement(range(4),d)]).T),('oddSelf',lambda x,j:np.c_[np.ones(len(x)),x,x[:,j]**2,x[:,j]**3,x[:,j]**5,x[:,j,None]*np.delete(x,j,axis=1),np.delete(x,j,axis=1)**2])]:
    check(name,design)
 rng=np.random.default_rng(8); folds=rng.permutation(len(X))%5
 for topology in [m._OPP,np.roll(np.arange(4),1),np.roll(np.arange(4),-1)]:
    m._OPP=topology
    e=[]
    for k in range(5):
      p=m._fit_calibration(x[folds[:len(x)]!=k],y[folds[:len(x)]!=k],b[folds[len(x):]!=k],z[folds[len(x):]!=k],v[folds[len(x):]!=k]); inds=folds[:len(x)]==k; e.extend(abs(m._readout(x[inds],p)-y[inds]).ravel())
    p=m._fit_calibration(x,y,b,z,v); print('trig',topology,np.quantile(e,[.5,.8,.95]),'fit',p)
 print('study metadata', s['nx'],s['ny'],s['robin_x'],s['robin_y'], 'diagnostics',type(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0].get('campaign_diagnostics')))
