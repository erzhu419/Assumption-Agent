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
from scipy.optimize import minimize_scalar
old=run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_02_solver.py'); x=old['_unwarp'](X,old['_calibrate'](s['panel_a']))
for name in ['dev_m101_c04_batch_03_solver.py','dev_m101_c04_batch_00_solver.py']:
 print(open('/tmp/hegel_continual_archive/'+name).read())
for out,fs in [('v1',{'abs-rat':lambda u,p:u*abs(u)/(1+p[0]*abs(u)),'sq-rat':lambda u,p:u*abs(u)/(1+p[0]*u*u),'power-rat':lambda u,p:u*abs(u)**p[1]/(1+p[0]*abs(u)**p[2]),'gate-rat':lambda u,p:np.sign(u)*np.maximum(abs(u)-p[1],0)**p[2]/(1+p[0]*abs(u)),'sqrt':lambda u,p:u*abs(u)/np.sqrt(1+p[0]*u*u)}),('v2',{'sq-rat':lambda u,p:u*u/(1+p[0]*u*u),'abs-rat':lambda u,p:u*u/(1+p[0]*abs(u)),'power-rat':lambda u,p:abs(u)**p[1]/(1+p[0]*abs(u)**p[2]),'sqrt':lambda u,p:u*u/np.sqrt(1+p[0]*u*u)})]:
 u=s['panel_c']['u']; v=s['panel_c'][out]; best=(np.inf,None)
 for name,g in fs.items():
  for mode in [0,1]:
   def D(z,p):return a(g(z,p) if mode==0 else np.c_[z,g(z,p)])
   def fitop(m):
    def err(p):
     F=D(u[m],p); return (F@ls(F,v[m])-v[m]).ravel()
    p0={'power-rat':[.3,1.,1.], 'gate-rat':[.3,.03,1.8]}.get(name,[.3]); low={'power-rat':[0,.1,.1],'gate-rat':[0,0,.3]}.get(name,[0]); hi={'power-rat':[10,4,5],'gate-rat':[10,.4,4]}.get(name,[10]);p=least_squares(err,p0,bounds=(low,hi),max_nfev=200).x; C=ls(D(u[m],p),v[m]); return p,lambda z:D(z,p)@C
   e=[]
   for k in range(4):
    m=np.arange(len(u))%4!=k; F=fitop(m)[1]; e.append(abs(F(u[~m])-v[~m]).mean())
   p,F=fitop(np.ones(len(u),bool)); print(out,name,mode,np.round(p,6),round(abs(F(u)-v).mean(),7),round(np.mean(e),7));
   if np.mean(e)<best[0]:best=(np.mean(e),F)
 if out=='v1' and best[0]<.001: c=best[1]
 if out=='v2' and best[0]<.00027: d=best[1]
z={'x':x,'b':b(x),'c':c(x),'d':d(x)}; z.update({k+j:f(z[j]) for k,f in [('b',b),('c',c),('d',d)] for j in 'bcd'})
t=s['eval_times']; H=-np.expm1(-t[:,None]*[.4,1.82]); V=np.c_[tr['y'],ls(H,tr['traj'].T).T]; idx=np.arange(72); folds=[(idx[:56],idx[56:])]+[(idx[idx%5!=i],idx[idx%5==i]) for i in range(5)]
for k,j in [('b','db'),('b','c'),('b','cd'),('b','d'),('b','bd'),('b','cb'),('c','d')]:
 u,w=z[k],z[j]; F=a(np.c_[u,w,u*u,w*w,u*w]); C=ls(F[:72],V); print(k,j,'coeff rows by family',C.T.round(4).tolist());
 for q in [5,9,13,17,21]:
  D=F[:,:q]; E=[]
  for i,h in folds:
   r=D[h]@ls(D[i],V[i])-V[h];E.append(np.r_[abs(r).mean(0),abs(r[:,1:]@H.T).mean()]);
  print(q,np.array(E)[0].round(5),np.max(E,axis=0).round(5))
