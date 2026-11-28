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
t=s['eval_times']; tbase=-np.expm1(-t[:,None]*[.4,1.82]); A=ls(tbase,tr['traj'].T).T; V=np.c_[tr['y'],A]
idx=np.arange(72); folds=[(idx[:56],idx[56:])]+[(idx[idx%5!=j],idx[idx%5==j]) for j in range(5)]
raw={'x':x,'raw':X,'b':b(x),'c':c(x),'d':d(x),'g':gate(x,[.35,1.5]),'h':x/(1+.83*abs(x)),'q':x*x/(1+.164*x*x)}
for k,op in [('b',b),('c',c),('d',d)]:
 for j in 'bcd':raw[k+j]=op(raw[j])
for k in ('b','c','d'):
 for j,p in [('raw',X),('direct',X*gain+offset)]:raw[k+j]={'b':b,'c':c,'d':d}[k](p)
def errors(pred,target):
 e=pred-target; return np.r_[np.mean(abs(e),0),np.mean(abs(e[:,1:]@tbase.T))]
def evaluate(D):
 E=np.array([errors(D[j]@ls(D[i],V[i]),V[j]) for i,j in folds]);return np.r_[E[0],E[1:].mean(0),E[1:].max(0)]
res=[]
def add(k,z):res.append((k,evaluate(z)))
for k,z in raw.items():
 add(k,a(z)); add(k+'^2',a(np.c_[z,z*z])); add(k+'tanh',a(np.c_[z,np.tanh(z)]))
for k,j in combinations(raw,2):
 z,w=raw[k],raw[j]; add(k+'+'+j,a(np.c_[z,w]));
 for f,F in [('diag',z*w),('sum',np.sum(z*w,axis=1)[:,None]),('all',(z[:,:,None]*w[:,None,:]).reshape(len(z),16))]:add(k+'+'+j+f,a(np.c_[z,w,F]))
for target in (4,5,6,7):
 print('target',target)
 for k,e in sorted(res,key=lambda r:r[1][target])[:7]: print(k,e.round(4))
# Sparse selection across both transformed coordinates and cross-branch interactions.
B={}
for k,z in raw.items():
 for j in range(4):
  for name,f in [('id',lambda z:z),('sq',np.square),('abs',np.abs),('tanh',np.tanh),('relu',lambda z:np.maximum(z,0))]:B[f'{name}({k}{j})']=f(z[:,j])
for k,j in combinations(list(raw)[:8],2):
 for l,m in product(range(4),repeat=2):B[f'{k}{l}*{j}{m}']=raw[k][:,l]*raw[j][:,m]
for k in list(raw)[:8]:
 for l,m in combinations(range(4),2):B[f'{k}{l}*{k}{m}']=raw[k][:,l]*raw[k][:,m]
for k,j in combinations('bcd',2):
 for l,m in product(range(4),repeat=2):
  for f,g in [('sum',np.add),('diff',np.subtract)]:
   v=g(raw[k][:,l],raw[j][:,m]); B[f'{f}({k}{l},{j}{m})sq']=v*v; B[f'{f}({k}{l},{j}{m})tanh']=np.tanh(v)
D=np.column_stack(list(B.values())); names=np.array(list(B));
for forced in [[],['b'],['b','c'],['b','c','d'],['x','b','c','d']]:
 print('forced',forced); E=[]
 for i,j in folds:
  F=a(np.column_stack([raw[k] for k in forced])) if forced else np.ones((len(x),1)); U=F[i]; R=D[i]-U@ls(U,D[i]); norm=np.maximum(np.linalg.norm(R,axis=0),1e-10); M=R/norm; M[:,norm<1e-7]=0; out=[]
  for k,v in enumerate(V.T):
   S=[]; rv=v[i]-U@ls(U,v[i]); preds=[]
   for h in range(1,16):
    k0=np.argmax(abs(M.T@rv)); S.append(k0); M[:,k0]=M[:,k0] # independent targets
    G=np.c_[U,D[i][:,S]]; co=ls(G,v[i]);rv=v[i]-G@co
    if h in (3,6,10,15):preds.append(np.c_[F[j],D[j][:,S]]@co)
   if np.array_equal(i, folds[0][0]):print(k,names[S[:8]].tolist(),np.mean(abs(rv)))
   out.append(preds)
  E.append([errors(p,V[j]) for p in np.array(out).transpose(1,2,0)])
 E=np.array(E);print('56/16',E[0].round(4),'mean',E[1:].mean(0).round(4),'worst',E[1:].max(0).round(4))
