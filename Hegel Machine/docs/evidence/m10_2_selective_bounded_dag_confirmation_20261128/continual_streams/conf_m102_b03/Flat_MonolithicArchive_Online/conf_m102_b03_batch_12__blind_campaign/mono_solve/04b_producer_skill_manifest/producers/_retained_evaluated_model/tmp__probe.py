import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
D=lambda x:np.c_[np.ones(len(x)),x]
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
a=s['panel_a']; X=D(a['u']); v=a['v']; rng=np.random.default_rng(1); p=min([ls(X[i],v[i]) for i in [rng.choice(len(v),8,False) for _ in range(300)]],key=lambda p:np.median(np.linalg.norm(X@p-v,axis=1))); r=np.linalg.norm(X@p-v,axis=1); good=r<max(.003,5*np.median(r)); p=ls(X[good],v[good]); print('A',sum(good),p,np.sort(np.linalg.norm(X@p-v,axis=1)))
z=(tr['x']-p[0])@np.linalg.inv(p[1:]); b=s['panel_b']; w=ls(D(b['u']),b['v']); print('B',w,'sing',np.linalg.svd(w[1:])[1]); h=lambda r:-np.expm1(-s['eval_times'][:,None]*r); t=tr['traj']; rates=least_squares(lambda r:(h(r)@ls(h(r),t[:n].T)-t[:n].T).ravel(),[2,.5]).x; amp=ls(h(rates),t.T).T; y=np.c_[tr['y'],amp]; print('rates',rates,'correlation',np.corrcoef(y.T),'amp sd',y.std(0)); print('latent ranges',z.min(0),z.max(0))
poly=lambda q,deg: np.column_stack([np.prod(q[:,I],axis=1) for j in range(deg+1) for I in combinations_with_replacement(range(q.shape[1]),j)])
def report(name,F):
 pred=F@ls(F[:n],y[:n]); print(name,F.shape[1],np.mean(abs(pred[:n]-y[:n]),0),np.mean(abs(pred[n:]-y[n:]),0))
for name,q in [('raw',tr['x']),('latent',z),('B',D(z)@w)]:
 for d in [1,2,3,4]:report(name+str(d),poly(q,d))
 for d,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('softplus',lambda x:np.logaddexp(0,x)),('softsign',lambda x:x/(1+abs(x)))]:
  for k in [1.25,2.15]:report(name+d+str(k),D(np.c_[q,f(k*q)]))
c=s['panel_c']; u=c['u']; print('C raw first',np.c_[u,c['v1'],c['v2']][:12]);print('C output corr',np.corrcoef(np.c_[c['v1'],c['v2']].T).round(2))
for name,q in [('u',u),('B',D(u)@w)]:
 for key in ['v1','v2']:
  v=c[key]; scores=[]
  for label,F in [('pol2',poly(q,2))]+[(label+str(k),D(np.c_[q,f(k*q)])) for label,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('softsign',lambda x:x/(1+abs(x))),('swish',lambda x:x/(1+np.exp(-x))),('relu',lambda x:np.maximum(0,x))] for k in [1.25,2.15,4,6,8]]:
   pin=np.linalg.pinv(F); e=(F@(pin@v)-v)/np.maximum(1-np.sum(F*pin.T,axis=1),1e-6)[:,None];scores.append((np.mean(e**2)**.5,label))
  print('C',name,key,sorted(scores)[:10])
np.savez('/tmp/core.npz', z=z,w=w,y=y,rates=rates,cal=p)
