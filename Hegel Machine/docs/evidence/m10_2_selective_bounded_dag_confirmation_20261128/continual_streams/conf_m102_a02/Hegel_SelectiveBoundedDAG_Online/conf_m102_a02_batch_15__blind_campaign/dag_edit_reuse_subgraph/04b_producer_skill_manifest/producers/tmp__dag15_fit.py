import numpy as np,pickle
from scipy.optimize import minimize_scalar,root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; U,V=(s['panel_a'][k] for k in ('u','v')); ok=np.max(abs(V),1)<3; U,V=U[ok],V[ok]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def feats(u,j,w,mode):
 a=u[:,j]; b=u[:,(j+2)%4]; t=np.sin(w*b)
 return np.c_[np.ones(len(a)),u,t,*({'s':[],'a':[a**3],'ab':[a**3,b**3],'q':[a*a,a*b,b*b,a**3],'all':[u**3],'cross':[a**3,a*a*b,a*b*b],'sine2':[np.sin(w*a)],'gated':[a**3,t*a*a,t*a] }[mode])]
for mode in ('s','a','ab','q','all','cross','sine2','gated'):
 def fit(u,v):
  cc=[]
  for j in range(4):
   def fn(w):
    f=feats(u,j,w,mode);return np.mean((f@ls(f,v[:,j])-v[:,j])**2)
   grid=np.linspace(.8,5,75); w=grid[np.argmin([fn(w) for w in grid])]; w=minimize_scalar(fn,bounds=(w-.07,w+.07),method='bounded').x; c=ls(feats(u,j,w,mode),v[:,j]); cc.append((w,c))
  return cc
 def pred(u,cc):return np.column_stack([feats(u,j,w,mode)@c for j,(w,c) in enumerate(cc)])
 cc=fit(U,V); r=pred(U,cc)-V; errs=[]
 for test in np.array_split(np.arange(len(U)),5):
  tr=np.setdiff1d(np.arange(len(U)),test); errs.extend((pred(U[test],fit(U[tr],V[tr]))-V[test]).ravel())
 print(mode,'rmse',np.sqrt(np.mean(r*r)),np.std(errs),[(round(w,6),np.round(c,5)) for w,c in cc],flush=True)
 if mode in ('a','sine2','gated','ab'):
  x=np.vstack([s['train']['x'],s['test']['x']]); qs=[]; rr=[]
  for row in x:
   fn=lambda q:pred(q[None,:],cc)[0]-row
   z=root(fn,row/1.2); qs.append(z.x);rr.append(np.linalg.norm(fn(z.x)))
  q=np.asarray(qs); u,v=(s['panel_b'][k] for k in ('u','v')); B=lambda q:np.c_[np.ones(len(q)),np.cos(2.6*q)];b=B(q)@ls(B(u),v); np.savez('/tmp/dag15_'+mode+'.npz',q=q,b=b,y=s['train']['y'],z=s['train']['traj']); print('inverse',np.max(rr),np.max(abs(q)))
