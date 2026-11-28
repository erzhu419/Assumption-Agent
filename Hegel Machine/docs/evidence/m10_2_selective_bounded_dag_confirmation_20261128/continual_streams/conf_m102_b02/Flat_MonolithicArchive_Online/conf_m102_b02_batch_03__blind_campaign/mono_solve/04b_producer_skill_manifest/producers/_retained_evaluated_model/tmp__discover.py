import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
def I(x):return np.c_[np.ones(len(x)),x]
def calbasis(x):return I(np.c_[x,x*np.roll(x,-1,axis=1),x**3])
p=s['panel_a']; mask=np.linalg.norm(p['v'],axis=1)<5
coef=ls(calbasis(p['u'][mask]),p['v'][mask]);coef[abs(coef)<.001]=0
x=np.r_[s['train']['x'],s['test']['x']];z=np.array([least_squares(lambda q:(calbasis(q[None,:])@coef)[0]-r,ls(coef[1:5].T,r-coef[0])).x for r in x]);p=s['panel_b'];bc=ls(I(p['u']),p['v']);w=I(z)@bc
p=s['panel_c'];u=p['u']; cands={}
for k in ['v1','v2']:
 v=p[k]; out=[]
 for name,fn in [('tanh',np.tanh),('erf',erf),('sin',np.sin),('asinh',np.arcsinh),('rat',lambda a:a/(1+a*a)),('atan',np.arctan),('silu',lambda a:a/(1+np.exp(-a))),('poly2',lambda a:a*a),('poly3',lambda a:a**3)]:
  for even in [0,1,2]:
   for lin in [0,1]:
    if name.startswith('poly') and (not lin or even):continue
    for shared in [True,False]:
     if name.startswith('poly') and not shared:continue
     def d(x,q):
      return np.column_stack([I(fn(x*np.exp(q)))]+([x] if lin else [])+([x*x] if even==1 else [x*x,x*np.roll(x,-1,axis=1)] if even==2 else []))
     def r(q):
      a=d(u,q);return (a@ls(a,v)-v).ravel()
     fit=min([least_squares(r,np.full(1 if shared else 4,np.log(f)),bounds=(-2,1.5),max_nfev=85) for f in (.45,1.1,2,3.45)],key=lambda f:f.cost)
     q=fit.x;a=d(u,q);c=ls(a,v);err=a@c-v;h=(a*np.linalg.pinv(a).T).sum(1);loo=np.sqrt(np.mean((err/(1-h[:,None]))**2));out.append((loo,name,even,lin,np.exp(q),d(z,q)@c,d(w,q)@c,np.sqrt(np.mean(err**2))))
 out.sort(key=lambda x:x[0]); cands[k]=out
 for a in out[:9]:print(k,a[:5],a[-1])
 np.savez('/tmp/'+k+'.npz',**{f'z{i}':r[5] for i,r in enumerate(out[:9])},**{f'w{i}':r[6] for i,r in enumerate(out[:9])})
np.savez('/tmp/coords.npz',z=z,w=w)
t=s['eval_times'];T=s['train']['traj'];
for name,fn in [('exp',lambda t,q:1-np.exp(-t*q)),('exp+t',lambda t,q:np.c_[1-np.exp(-t*q),t]),('exp+sq',lambda t,q:np.c_[1-np.exp(-t*q),(1-np.exp(-t*q))**2]),('exp+sin',lambda t,q:np.c_[1-np.exp(-t*q[0]),np.sin(t*q[1])]),('exp+exp',lambda t,q:1-np.exp(-t[:,None]*q))]:
 def r(q):
  a=fn(t,q).reshape(len(t),-1);return (a@ls(a,T.T)-T.T).ravel()
 f=min([least_squares(r,np.array(q),bounds=(.001,5)) for q in ([(.2,),(.8,),(1.2,)] if name not in ['exp+sin','exp+exp'] else [(.4,1.2),(.8,1.25),(1.2,2.5)])],key=lambda f:f.cost);print(name,f.x,np.sqrt(np.mean(f.fun**2)))
