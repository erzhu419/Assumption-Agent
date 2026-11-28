import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def rms(x): return np.mean(x*x)**.5
p=s['panel_a']; u,v=p['u'],p['v']; cal=[]
for x,y in zip(u.T,v.T):
 i,j=np.triu_indices(len(x),1); m=np.median((y[i]-y[j])/(x[i]-x[j])); r=abs(y-m*x-np.median(y-m*x)); k=r<6*np.median(r); cal.append(ls(aff(x[k,None]),y[k]))
o,g=np.array(cal).T
z=(s['train']['x']-o)/g
p=s['panel_c']; u=p['u']; t=s['eval_times']; Y=s['train']['traj']; y=s['train']['y']
def branch(fn,v,q,bounds):
 def err(q):
  f=aff(fn(u,*q)); return (f@ls(f,v)-v).ravel()
 fit=least_squares(err,q,bounds=bounds,gtol=1e-13,ftol=1e-13,xtol=1e-13); F=aff(fn(u,*fit.x)); M=ls(F,v); print('branch',fit.x,rms(fit.fun),'matrix',M,'singular',np.linalg.svd(M[1:],compute_uv=False)); return aff(fn(z,*fit.x))@M
from scipy.special import erf
v=p['v1']; x=u; xx=abs(x); I=np.arange(len(u)); folds=[(I[I%4!=j],I[I%4==j]) for j in range(4)]
def prof(F,i):
 f=F[i]; V=v[i]; a=f-f.mean(0); b=V-V.mean(0); h,d,j=np.linalg.svd(a.T@b); m=(h@j)*d.sum()/np.sum(a*a); return (F-f.mean(0))@m+V.mean(0)-v
specs=[('erf',lambda a,k:a*x+erf(k*x),[.5,2],[0,.1],[20,30]),('expsign',lambda a,k:a*x+np.sign(x)*(-np.expm1(-k*xx)),[.5,2],[0,.1],[20,30]),('abs-softsign',lambda a,k:a*x+x/(1+k*xx),[.5,2],[0,.01],[20,80]),('general-softsign',lambda a,k,b,c:a*x+x/(1+k*xx**b)**c,[.5,2,1,.8],[0,.01,.2,.05],[20,80,8,8]),('power',lambda a,k:a*x+np.sign(x)*xx**k,[.5,.4],[0,.01],[20,.99]),('general-exp',lambda a,k,b:a*x+np.sign(x)*(-np.expm1(-k*xx**b)),[.5,2,1],[0,.01,.1],[20,80,8]),('general-tanh',lambda a,k,b:a*x+np.sign(x)*np.tanh(k*xx**b),[.5,2,1],[0,.01,.1],[20,80,8]),('asinh',lambda a,k:a*x+np.arcsinh(k*x),[.1,2],[-10,.01],[20,80]),('rational4',lambda a,k,b:a*x+x/(1+k*x**2+b*x**4),[.5,2,.1],[0,.01,-.1],[20,80,80]),('rational-abs',lambda a,k,b:a*x+x/(1+k*xx+b*x**2),[.5,2,.1],[0,0,0],[20,80,80]),('two-tanh',lambda a,k,b,c:a*x+np.tanh(k*x)+b*np.tanh(c*x),[.5,4,.5,1],[0,.1,-4,.1],[20,40,8,40]),('atan+tanh',lambda a,k,b,c:a*x+np.arctan(k*x)+b*np.tanh(c*x),[.5,4,.5,1],[0,.1,-4,.1],[20,40,8,40]),('atan+lin','',[],[],[]),('tanh+rat',lambda a,k,b,c:a*x+np.tanh(k*x)+b*x/(1+c*x*x),[3,6,10,1],[0,.1,-20,.01],[20,40,50,80]),('sin+atan',lambda a,k,b,c:a*x+np.arctan(k*x)+b*np.sin(c*x),[.25,4,.1,2],[-20,.01,-10,.01],[20,40,10,30]),('linear-sin',lambda a,k:x+a*np.sin(k*x),[.5,3],[0,.1],[10,20]),('linear-clip',lambda a,k:x+a*np.clip(x,-k,k),[.5,.4],[0,.01],[10,3]),('sigmoid-shift-scaled',lambda a,k,b,c:a*x+(np.tanh(k*(x-b))+np.tanh(k*(x+b)))*.5+c*x**3,[.5,2.5,.1,0],[0,.1,0,-2],[20,40,1,2])]
for name,fn,q,lo,hi in specs:
 if not q: continue
 def fit(i,q): return least_squares(lambda a:prof(fn(*a),i)[i].ravel(),q,bounds=(lo,hi),max_nfev=400,ftol=1e-11,gtol=1e-11)
 f=fit(I,q); cv=[rms(prof(fn(*fit(i,f.x).x),i)[j]) for i,j in folds]; print(name,np.round(f.x,7),round(rms(f.fun),8),round(max(cv),8))
