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
ops={}; bank={'x':x}
for tag,panel,out in [('b','panel_b','v'),('c','panel_c','v1'),('d','panel_c','v2')]:
 u=s[panel]['u'];v=s[panel][out]; scores=[]
 for k in lib:
  g,p,e=fit(u,v,k); errs=[]
  for j in range(4):
   m=np.arange(len(u))%4!=j; gg,_,_=fit(u[m],v[m],k);errs.append(np.mean(abs(gg(u[~m])-v[~m])))
  scores.append((np.mean(errs),k,g))
  print(tag,k,'%.6f %.6f'%(e,np.mean(errs)),p)
 scores.sort(key=lambda t:t[0]); ops[tag]=scores[0][2]
 for _,k,g in scores[:3]:bank[tag+k]=g(x)
for i,g in ops.items():
 bank[i]=g(x)
 for j,h in ops.items():bank[j+i]=h(g(x))
base={k:bank[k] for k in 'xbcd'}
for i,z in base.items():
 for j,w in base.items():
  if i<j:bank[i+'_'+j]=np.c_[z,w]
idx=np.arange(72); folds=[(idx[:56],idx[56:])]+[(idx[idx%5!=j],idx[idx%5==j]) for j in range(5)]
res=[]
for k,z in bank.items():
 cs=(z[:,0]*z[:,2]+z[:,1]*z[:,3])[:,None]
 for label,D in [('linear',a(z)),('cross',a(np.c_[z,cs])),('sep2',sep(z,2)),('poly2',poly(z,2)),('satur',a(np.c_[z,np.tanh(z)]))]:
  if D.shape[1]>46:continue
  E=[]
  for i,j in folds:
   e=abs(D[j]@ls(D[i],Y[i])-Y[j]);E.append([e[:,0].mean(),e[:,1:].mean()])
  E=np.asarray(E);res.append((k,label,*E[0],*E[1:].mean(0),*E[1:].max(0)))
for col in [4,5]:
 print('rank',col,'56/16 y, state; avg y,state; worst y,state');
 for r in sorted(res,key=lambda v:v[col])[:18]:print(r[:2],np.round(r[2:],5))
