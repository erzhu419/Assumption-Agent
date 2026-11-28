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
ops={}
for tag,out in [('c','v1'),('d','v2')]:
 u=s['panel_c']['u'];v=s['panel_c'][out]; rows=[]
 for kind in forms:
  for lin in (False,True):
   for shift in (False,True):
    e,p,C,f=search(u,v,kind,lin,shift); rows.append((e,kind,lin,shift,p,C,f))
 for _,kind,lin,shift,p,C,f in sorted(rows,key=lambda r:r[0])[:9]:
  es=[]
  for i in range(4):
   m=np.arange(len(u))%4!=i; ff=search(u[m],v[m],kind,lin,shift)[-1];es.append(np.mean(abs(ff(u[~m])-v[~m])))
  print(tag,kind,lin,shift,'fit',_, 'cv',np.mean(es),'p',p,'coef',C.round(4).tolist())
 ops[tag]=min(rows,key=lambda r:r[0])[-1]
b,p,e=fit(s['panel_b']['u'],s['panel_b']['v'],'gate'); ops['b']=b
print('B fit',e,'params',p,'coefs',ls(a(gate(s['panel_b']['u'],p)),s['panel_b']['v']).round(4))
print('telemetry',s['telemetry_vec'],'summary',tr['summary']); print('y range',Y[:,0].min(),Y[:,0].max())
t=s['eval_times'];T=tr['traj'];
def timebasis(p):return -np.expm1(-t[:,None]*p)
def r(p):
 h=timebasis(p);return (h@ls(h,T.T)-T.T).ravel()
q=least_squares(r,[.4,1.8],bounds=(.01,30.)).x
print('rates',q,'MAE',np.mean(abs(r(q)))); amp=ls(timebasis(q),T.T).T; print('amplitude quantiles',np.quantile(amp,[0,.25,.5,.75,1],axis=0))
for tag,g in {'x':lambda x:x,**ops}.items():
 z=g(x[:72]); print(tag,'corr rows x-/plus-amp-and-y',np.corrcoef(np.c_[z,amp,Y[:,0]].T)[:4,4:].round(3).tolist())
for i in range(15):print(i,np.round(x[i],3),np.round(amp[i],3),round(Y[i,0],3))
