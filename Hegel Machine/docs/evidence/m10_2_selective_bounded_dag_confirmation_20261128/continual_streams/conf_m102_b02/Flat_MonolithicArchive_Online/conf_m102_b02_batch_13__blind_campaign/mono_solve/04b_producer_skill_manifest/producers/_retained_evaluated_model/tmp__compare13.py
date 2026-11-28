import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; I=m._intercept; ls=m._ls; z=m._calibrate(s['panel_a'],t['x']); y=np.c_[t['y'],t['traj']]; rng=np.random.default_rng(709); splits=[np.arange(len(z))]+[rng.permutation(len(z)) for _ in range(32)]+[np.argsort(np.linalg.norm(z,axis=1))]+[np.argsort(sign*z[:,j]) for j in range(4) for sign in (-1,1)]; p=s['panel_b']; c=s['panel_c']
def design(h):return I(np.c_[z,h,(z[:,:,None]*h[:,None,:]).reshape(len(z),16)])
def score(d,reg=0):
 a=[]
 for p in splits:
  i,j=p[:52],p[52:]; _,_,v=np.linalg.svd(y[i,1:],full_matrices=False); target=np.c_[y[i,0],y[i,1:]@v[:2].T]; x=d[i]; coef=ls(x,target) if reg==0 else np.linalg.solve(x.T@x+np.diag(np.r_[0,np.full(x.shape[1]-1,reg)]),x.T@target); pred=d[j]@coef; e=abs(np.c_[pred[:,0],pred[:,1:]@v[:2]]-y[j]); a.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()]);
 return np.round(a[0],6),np.round(np.max(a,axis=0),6)
def csoft(a,q):return I(np.c_[a,a/(1+q*abs(a))])
w=m._operator(p,'v',lambda a,q:I(np.cos(q*a)),2.6,(1.5,3.5),z)
for name,b in [('cos',w),('identity',z),('sine',m._operator(p,'v',lambda a,q:I(np.c_[a,np.sin(q*a)]),10.87,(9,13),z)),('even rational',m._operator(p,'v',lambda a,q:I(1/(1+(q*a)**2)),1.16,(.3,3),z))]:
 h=m._operator(c,'v1',csoft,.83,(.2,2.5),b); print('B ablation',name,score(design(h)))
for name,fn,q0 in [('baseline',lambda a,q: csoft(a,q[0]),[.83]),('odd quotient',lambda a,q:I(np.c_[a,a/(1+q[0]*abs(a)+q[1]*a*a)]),[.93,.34]),('linear-plus-squared-softsign',lambda a,q:I(np.c_[a,a/(1+q[0]*abs(a))**2]),[.6]),('linear-plus-cubed-softsign',lambda a,q:I(np.c_[a,a/(1+q[0]*abs(a))**3]),[.32])]:
 u,v=c['u'],c['v1']; fit=least_squares(lambda q: (fn(u,q)@ls(fn(u,q),v)-v).ravel(),q0,bounds=(.05,4)); h=fn(w,fit.x)@ls(fn(u,fit.x),v); d=design(h); print('C and penalty sweep',name,fit.x,np.std(fit.fun));
 b=ls(d[:52],y[:52]); print('y bilinear singular values',np.linalg.svd(b[9:,0].reshape(4,4),compute_uv=False));
 for lam in [0,.00001,.0001,.001,.005,.02]: print(lam,score(d,lam))
