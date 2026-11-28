import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; v=p['v2']; t=s['eval_times']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; n=len(x)
def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def cross(x):return x[:,[0,0,1,2]]*x[:,[1,3,2,3]]
def B(x):return np.c_[np.ones(len(x)),cross(x)]
def pair(x,y):return (x[:,:,None]*y[:,None,:]).reshape(len(x),-1)
models=[('cos',lambda x,k:x+k[0]*(1-np.cos(k[1]*x)),[.66,1.27]),('cos-odd',lambda x,k:x+k[0]*(1-np.cos(k[1]*x))+k[2]*(np.sin(k[1]*x)-k[1]*x),[.66,1.27,0]),('logcosh',lambda x,k:x+k[0]*np.log(np.cosh(k[1]*x)),[1,.9]),('even-rational',lambda x,k:x+k[0]*x*x/(1+k[1]*x*x),[.55,.16]),('even-exp',lambda x,k:x-k[0]*np.expm1(-k[1]*x*x),[1.6,.32]),('sin-square',lambda x,k:x+k[0]*np.sin(k[1]*x*x),[.76,.7]),('sqrt',lambda x,k:x+k[0]*(np.sqrt(1+k[1]*x*x)-1),[1,.8]),('even-rational2',lambda x,k:x+k[0]*x*x/(1+k[1]*x*x+k[2]*x**4),[.55,.16,0]),('cos-coordinate',lambda x,k:x+k[:4]*(1-np.cos(k[4:]*x)),[.66]*4+[1.27]*4)]
for name,fn,init in models:
 def residual(k):
  W=aff(fn(u,k)); return (W@fit(W,v)-v).ravel()
 opt=least_squares(residual,init,max_nfev=250); print('v2',name,np.round(opt.x,6),round(np.std(opt.fun),8))
 if name=='cos': cos=lambda q:fn0(q,k0); fn0,k0=fn,opt.x
pb=s['panel_b']; f=aff(x)@fit(aff(pb['u']),pb['v']); g=B(x)@fit(B(u),p['v1']); h2=np.c_[aff(x),x*x]@fit(np.c_[aff(u),u*u],v); hc=aff(cos(x))@fit(aff(cos(u)),v)
L,R=np.triu_indices(4); Q=np.c_[aff(f),f[:,L]*f[:,R]]; Y=np.c_[y,z]
folds=[np.arange(52,n),*np.array_split(np.random.default_rng(42).permutation(n),5),np.argsort(np.linalg.norm(x,axis=1))[-16:]]
for name,X in [('v1+v2poly',np.c_[Q,f*g,pair(f,h2)]),('v1+v2cos',np.c_[Q,f*g,pair(f,hc)]),('v1+both',np.c_[Q,pair(f,g),pair(f,h2),pair(f,hc)])]:
 errors=[]; print(name,'rank/columns',np.linalg.matrix_rank(X),X.shape[1]);
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); A=X[tr]; C=X[te]; pred=C@fit(A,Y[tr]);
  def T(k):return -np.expm1(-t[:,None]*np.sort(k))
  def res(k):
   W=T(k); return (fit(W,z[tr].T).T@W.T-z[tr]).ravel()
  k=least_squares(res,[.41,1.8],bounds=(.05,5.)).x; W=T(k); amp=fit(W,z[tr].T).T
  temporal=C@fit(A,amp)@W.T
  e=abs(pred-Y[te]); r=abs(temporal-z[te]); errors.append([e[:,0].mean(),e[:,1:].mean(),r.mean(),r.mean(0).max()]);
 print(np.round(np.array(errors),5))
