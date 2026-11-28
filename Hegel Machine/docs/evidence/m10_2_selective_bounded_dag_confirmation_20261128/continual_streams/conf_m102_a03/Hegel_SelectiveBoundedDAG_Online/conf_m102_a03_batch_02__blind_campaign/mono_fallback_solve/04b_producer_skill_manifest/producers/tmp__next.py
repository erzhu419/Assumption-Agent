exec(open('/tmp/probe.py').read().split("print('times'")[0])
from conf_m102_a03_batch_01_solver import _inverse
from scipy.optimize import least_squares
pa=s['panel_a']; ua,va=pa['u'],pa['v']; fs=P(ua,2); c=fit(fs,va)
for _ in range(30):
 r=np.linalg.norm(fs@c-va,axis=1); wt=np.sqrt(np.minimum(1,1.5*np.median(r)/np.maximum(r,1e-12))); c=fit(fs*wt[:,None],va*wt[:,None]);
good=r<5*np.median(r); print('A good',good.sum());
for d in (1,2,3):
 f=P(ua,d); c=fit(f[good],va[good]); print('A',d,np.std((f@c-va)[good]),c.T)
vv=np.roll(ua,-1,axis=1); aa=np.stack([np.ones_like(ua),ua,vv,ua*vv,ua**3],axis=1); print('old A residual',np.sort(np.linalg.norm(np.einsum('ijk,jk->ik',aa,a)-va,axis=1))); print('old A coef',a)
qs={'x':x,'Ainv':_inverse(x,a),'Afor':np.einsum('ijk,jk->ik',np.stack([np.ones_like(x),x,np.roll(x,-1,1),x*np.roll(x,-1,1),x**3],axis=1),a)}
for key in list(qs): qs[key+' B']=A(qs[key])@b
for key,q in qs.items():
 print('\nMODEL',key)
 for deg in (1,2,3,4,5):
  for mode in (['full'] if deg<=2 else ['pure','full'] if deg==3 else ['pure']):
   F=P(q,deg) if mode=='full' else np.c_[A(q),*[q**i for i in range(2,deg+1)]]
   if np.max(np.abs(F))>1e6: continue
   c=fit(F[:n],y[:n]); print(deg,mode,np.mean(abs((F@c-y)[:n])),np.mean(abs((F@c-y)[n:])),c if deg<=3 else '')
 print('traj',np.mean(abs((A(q)@fit(A(q)[:n],t[:n])-t)[n:])))
