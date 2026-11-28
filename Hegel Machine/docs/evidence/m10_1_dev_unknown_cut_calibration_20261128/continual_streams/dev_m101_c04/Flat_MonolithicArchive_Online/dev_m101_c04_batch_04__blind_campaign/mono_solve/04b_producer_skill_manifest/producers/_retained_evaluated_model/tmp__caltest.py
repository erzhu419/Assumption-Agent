import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from runpy import run_path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=np.r_[tr['x'],s['test']['x']]; Y=np.c_[tr['y'],tr['traj']]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
a=lambda x:np.c_[np.ones(len(x)),x]
cal=run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_03_solver.py'); offset,gain=cal['_calibrate'](s['panel_a']); x=(X-offset)/gain
print('offset,gain',offset,gain)
def sep(u,d):return np.c_[a(u),*[u**j for j in range(2,d+1)]]
def poly(u,d):return np.c_[np.ones(len(u)),*[np.prod(u[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(u.shape[1]),k)]]
def gate(u,p):return np.sign(u)*np.maximum(abs(u)-p[0],0)**p[1]
lib={'linear':lambda u,p:a(u),'sep2':lambda u,p:sep(u,2),'sep3':lambda u,p:sep(u,3),'sep4':lambda u,p:sep(u,4),'poly2':lambda u,p:poly(u,2),'abs':lambda u,p:a(np.c_[u,abs(u)]),'gate':lambda u,p:a(gate(u,p)),'lgate':lambda u,p:a(np.c_[u,gate(u,p)]),'tanh':lambda u,p:a(np.tanh(u*p[0])),'ltanh':lambda u,p:a(np.c_[u,np.tanh(u*p[0])]),'rational':lambda u,p:a(np.c_[u,u/(1+p[0]*u*u)]),'sqrat':lambda u,p:a(np.c_[u,u*u/(1+p[0]*u*u)]),'labs3':lambda u,p:a(np.c_[u,abs(u),u*abs(u),u**3]),'sincos':lambda u,p:a(np.c_[u,np.sin(p[0]*u),np.cos(p[0]*u)])}
pars={'gate':([.3,1.5],[0,.5],[1,5]),'lgate':([.3,1.5],[0,.5],[1,5]),'tanh':([1],[.1],[10]),'ltanh':([1],[.1],[10]),'rational':([1],[.001],[20]),'sqrat':([1],[.001],[20]),'sincos':([1],[.1],[8])}
def fit(u,v,k):
 f=lib[k]; p=[]
 if k in pars:
  start,low,high=pars[k]
  def res(p):
   D=f(u,p); return (D@ls(D,v)-v).ravel()
  p=least_squares(res,start,bounds=(low,high),max_nfev=60).x
 D=f(u,p); c=ls(D,v);return lambda z:f(z,p)@c, p, np.mean(abs(D@c-v))
from scipy.special import erf
from scipy.optimize import minimize_scalar
forms={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'erf':erf,'softsign':lambda x:x/(1+abs(x)),'sqrtsoft':lambda x:x/np.sqrt(1+x*x),'rat':lambda x:x/(1+x*x),'clip':lambda x:np.clip(x,-1,1),'log':lambda x:np.sign(x)*np.log1p(abs(x)),'gauss':lambda x:-np.expm1(-x*x),'sqrtrise':lambda x:np.sqrt(1+x*x)-1,'sqrat':lambda x:x*x/(1+x*x),'cos':lambda x:1-np.cos(x),'sin':np.sin,'atan2':lambda x:np.arctan(x*x),'log2':lambda x:np.log1p(x*x),'elu':lambda x:np.expm1(np.minimum(x,0))+np.maximum(x,0)}
def search(u,v,kind,linear,shift=False):
 g=forms[kind]
 def design(x,p):
  G=g(p[0]*x+(p[1] if shift else 0)); return a(np.c_[x,G] if linear else G)
 def err(p):
  D=design(u,p); return (D@ls(D,v)-v).ravel()
 p=least_squares(err,[1.,0.] if shift else [1.],bounds=([.03,-2] if shift else [.03],[8.,2] if shift else [8.]),max_nfev=90).x
 C=ls(design(u,p),v); return np.mean(abs(err(p))),p,C,lambda x:design(x,p)@C
from itertools import combinations,product
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
b=fit(s['panel_b']['u'],s['panel_b']['v'],'gate')[0]
u=s['panel_c']['u']; c=search(u,s['panel_c']['v1'],'softsign',True)[-1]; d=fit(u,s['panel_c']['v2'],'sqrat')[0]
from itertools import combinations
A=s['panel_a']; U=A['u']; W=A['v']; rng=np.random.default_rng(19)
old=run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_02_solver.py')
co=old['_calibrate'](A); print('cyclic calibration coefficients',co)
for name,F in [('diag',[a(U[:,j:j+1]) for j in range(4)]),('affine',[a(U)]*4),('cyclic',[old['_calibration_features'](U,j) for j in range(4)]),('sep3',[sep(U,3)]*4),('affine_diag_nl',[a(np.c_[U,U[:,j]**2,U[:,j]**3]) for j in range(4)])]:
 pred=np.empty_like(W); counts=[]
 for j,D in enumerate(F):
  err=np.full(len(U),np.inf)
  for _ in range(180):
   ii=rng.choice(len(U),D.shape[1]+4,replace=False); e=abs(D@ls(D[ii],W[ii,j])-W[:,j]);
   if np.median(e)<np.median(err):err=e
  m=err<max(5*np.median(err),.0005); counts.append(m.sum()); pred[:,j]=D@ls(D[m],W[m,j]);
 print(name,'inlier median',np.median(abs(pred-W),axis=0),'counts',counts)
print('cyclic max correction',abs(old['_unwarp'](X,co)-x).max(0)); xx=old['_unwarp'](X,co)
t=s['eval_times']; T=tr['traj']; idx=np.arange(72); folds=[(idx[:56],idx[56:])]+[(idx[idx%5!=i],idx[idx%5==i]) for i in range(5)]
for calib,z0 in [('affine',x),('cyclic',xx),('raw',X)]:
 z={'x':z0,'b':b(z0),'c':c(z0),'d':d(z0)}
 z.update({k+j:f(z[j]) for k,f in [('b',b),('c',c),('d',d)] for j in 'bcd'})
 res=[]
 for k,j in combinations(z,2):
  v,w=z[k],z[j]
  for label,D in [('lin',a(np.c_[v,w])),('diag2',a(np.c_[v,w,v*v,w*w,v*w])),('sep2',sep(np.c_[v,w],2)),('poly2',poly(np.c_[v,w],2)),('tanh',a(np.c_[v,w,np.tanh(v),np.tanh(w)]))]:
   E=[]
   for i,h in folds:
    mu=D[i].mean(0); scale=np.maximum(D[i].std(0),1e-8); F=(D-mu)/scale; F[:,0]=1
    lam=0.1 if D.shape[1]<30 else 1.; C=np.linalg.solve(F[i].T@F[i]+np.diag(np.r_[0,np.full(D.shape[1]-1,lam)]),F[i].T@Y[i]); e=abs(F[h]@C-Y[h]); E.append([e[:,0].mean(),e[:,1:].mean()])
   E=np.asarray(E); res.append((k,j,label,*E[0],*E[1:].mean(0),*E[1:].max(0)))
 for p in (5,6):
  print(calib,p,[(r[:3],np.round(r[3:],4).tolist()) for r in sorted(res,key=lambda r:r[p])[:5]])
