import sys
sys.path.insert(0,'/holdout_task')
from node_00 import _inverse,_panel_b,_polys
from panels import *
from scipy.optimize import least_squares
from scipy.linalg import solve
p=s['panel_b']; U,V=p['u'],p['v']; u=_inverse(s['panel_a'],s['train']['x']); print('latent/panel absmax',abs(u).max(0),abs(U).max(0)); print('keys',s.keys()); print('B sorted per axis',np.sort(abs(U),axis=0)[-3:]); Xs={}
lin=lambda w:np.c_[np.ones(len(w)),w]
for name, fn, init, lo, hi in [('rational',lambda w,a:w**3/(1+a*w*w),.8,.001,8),('signedlog',lambda w,a:np.sign(w)*np.log1p(a*abs(w)**3),1.3,.001,8),('cubic_tanh',lambda w,a:np.tanh(a*w)**3,.75,.001,3),('sin',lambda w,a:np.sin(a*w),2.5,.05,6),('cubic',lambda w,a:w**3,1,.001,8),('sin3diff',lambda w,a:w-np.sin(a*w)/a,2.5,.05,6),('tanh',lambda w,a:np.tanh(a*w),1.3,.05,6),('erf',lambda w,a:__import__('scipy.special',fromlist=['erf']).erf(a*w),1.3,.05,6)]:
 for size in (1,4):
  for extra in ('linear','none','cross','square'):
   def F(w,th):return np.column_stack([np.ones(len(w)),fn(w,th)]+dict(none=[],linear=[w],cross=[features(w)['cross']],square=[w*w])[extra])
   def err(th):
    M=F(U,th); H=np.linalg.pinv(M); return ((V-M@(H@V))/np.maximum(.02,1-(M*H.T).sum(1))[:,None]).ravel()
   a=least_squares(err,np.repeat(init,size),bounds=(lo,hi),max_nfev=65); sc=np.sqrt(np.mean(err(a.x)**2)); c=np.linalg.lstsq(F(U,a.x),V,rcond=None)[0]; Xs[(name,size,extra)]=(sc,F(u,a.x)@c,a.x,c)
print('best B:');
for n,(sc,b,a,c) in sorted(Xs.items(),key=lambda n:n[1][0])[:9]: print(n,round(sc,6),a.round(4))
a=_panel_b(p,u); Xs['baseline']=(0,a,[],[]); z=s['train']['traj']; y=s['train']['y'];t=s['eval_times'];q=-np.expm1(-minimize_scalar(lambda k:np.linalg.norm(z-np.outer(z@(-np.expm1(-k*t))/sum(np.expm1(-k*t)**2),-np.expm1(-k*t))),bounds=(.1,2),method='bounded').x*t); amp=z@q/(q@q);Y=np.c_[y,amp]; print('profile',q,'amplitude correlation',np.corrcoef(Y.T)[0,1]);
from sklearn.model_selection import KFold
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=0).split(u)); opts=[]
from itertools import product
for k,(_,b,_,_) in Xs.items():
 if k!='baseline' and Xs[k][0]>.015:continue
 for family in ('poly','rational','pairs','sat'):
  for deg in (1,2,3):
   P=_polys(b,deg)
   for par in ([.1,.3,.7,1.5] if family!='poly' else [1.]):
    S = (b[:,i]+b[:,j])**2; D=(b[:,i]-b[:,j])**2
    if family=='poly': F=P
    if family=='rational': F=np.c_[P,(P[:,:,None]/(1+par*np.c_[S,D,b*b][:,None,:])).reshape(len(P),-1)]
    if family=='pairs':F=np.c_[P,(P[:,:,None]/(1+par*S[:,None,:])).reshape(len(P),-1)]
    if family=='sat':F=np.tanh(par*P)/par
    for lam in (1e-4,.001,.01,.1):
     es=[]
     for tr,te in folds:
      T=F[tr]; G=F[te]; m=T.mean(0); T=T-m;G=G-m; sc=np.maximum(.3,T.std(0));T=T/sc;G=G/sc
      av=Y[tr].mean(0); v=G@T.T@solve(T@T.T+lam*np.eye(len(tr)),Y[tr]-av,assume_a='pos')+av
      es.append(np.mean(abs(v-Y[te]),0)*[1,q.mean()])
     e=np.array(es);opts.append((k,family,deg,par,lam,e[0,0],e[0,1],*e[1:].max(0)))
for j in (5,6,7,8):print('best',j,sorted(opts,key=lambda x:x[j])[:5]);
print('balanced',sorted(opts,key=lambda x:max(x[5:7])+.3*max(x[7:]))[:7])
