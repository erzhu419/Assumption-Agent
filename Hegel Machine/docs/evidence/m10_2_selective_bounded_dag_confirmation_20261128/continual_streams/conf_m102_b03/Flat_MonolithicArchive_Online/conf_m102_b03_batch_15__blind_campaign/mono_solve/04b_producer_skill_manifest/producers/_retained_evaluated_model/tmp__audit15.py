import sys,pickle,importlib.util
import numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; u=s['panel_a']['u']; v=s['panel_a']['v']; w=np.roll(u,-1,axis=1)
print('study fields',sorted(s),'train fields',sorted(tr),'telemetry shape',s['telemetry_vec'].shape, 'target dtype',y.dtype)
# Identify gross panel contamination independently of the candidate basis.
g=np.argsort(np.linalg.norm(v-u,axis=1))
def features(pair):
    a=[np.ones_like(u),u,w,u**3];return np.stack(a+([u*w] if pair else []),axis=-1)
A=features(True);rng=np.random.default_rng(918)
def fit(a, rows):return np.stack([m._ls(a[rows,j],v[rows,j]) for j in range(4)])
def residual(a,c):return np.einsum('ijk,jk->ij',a,c)-v
pilots=[fit(A,rng.choice(len(u),8,False)) for _ in range(96)]
c=min(pilots,key=lambda c:np.median(np.max(np.abs(residual(A,c)),axis=1)))
g=np.flatnonzero(np.max(np.abs(residual(A,c)),axis=1)<.001)
for pair in (False,True):
    a=features(pair); E=np.array([residual(a,fit(a,g[g!=i]))[i] for i in g]);print('calibration leave-one-inlier-out pair',pair,'n',len(g),'MAE',np.mean(abs(E)),'max',abs(E).max())
z=m._calibrate(s['panel_a'],tr['x']); p,q=m._branches(s['panel_c'],z); dot=(p*q).sum(1); cross=p[:,0]*q[:,1]
S=m._design(np.c_[p,q,dot]); F=np.c_[S,cross]; Q=m._design(np.c_[q,cross]); dif=p-q; plus=p+q
# Extra linear sums and differences are already in span; test nonlinear combinations separately.
print('linear pair sum/difference nesting error',np.max(abs(S@m._ls(S,np.c_[plus,dif])-np.c_[plus,dif])))
aug={'base':np.empty((len(y),0)),'pair_squared':np.c_[dif**2,plus**2], 'rational':np.c_[p/(1+.22*q*q),q/(1+.22*p*p)], 'triple':np.c_[dot[:,None]*p,dot[:,None]*q]}
folds=[np.arange(len(y)),*[rng.permutation(len(y)) for _ in range(10)],np.argsort(np.linalg.norm(tr['x'],axis=1)),np.argsort(np.linalg.norm(tr['x'],axis=1))[::-1]]
for name,E in aug.items():
    R=[]
    for idx in folds:
        a,b=idx[:52],idx[52:]; ix=np.r_[a,b]; S1=np.c_[S,E]; F1=np.c_[F,E]; Q1=np.c_[Q,E]; yh=S1[b]@m._ls(S1[a],y[a]); st=m._predict_trajectory(t,T[a],F1[ix],Q1[ix]); e=np.abs(st-T[b]);R.append([np.mean(abs(yh-y[b])),e.mean(),e.mean(0).max()])
    print(name,'prescribed',np.round(R[0],7),'mean',np.round(np.mean(R,axis=0),7),'worst',np.round(np.max(R,axis=0),7))
# Compare time-only shared relaxation span with noise level to check for a missing dynamical mode.
from scipy.optimize import least_squares
H=lambda r:-np.expm1(-t[:,None]*r)
def er(r):h=H(r);return (h@m._ls(h,T[:52].T)-T[:52].T).ravel()
r=least_squares(er,[1.8,.42]).x; h=H(r);print('relaxation rates',r,'held-out projection noise MAE',np.mean(abs(h@m._ls(h,T[52:].T)-T[52:].T)))
