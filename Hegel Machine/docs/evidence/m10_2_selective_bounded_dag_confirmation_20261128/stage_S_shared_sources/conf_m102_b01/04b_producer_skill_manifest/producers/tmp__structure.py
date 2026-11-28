import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
u,v=s['panel_a'].values(); la=ls(aff(u),v)
for j in range(10):
 e=np.max(abs(aff(u)@la-v),axis=1); ii=np.argsort(e)[:39];la=ls(aff(u[ii]),v[ii]);
ii=np.max(abs(aff(u)@la-v),axis=1)<.025; la=ls(aff(u[ii]),v[ii]); print('cal',sum(ii),la)
b=s['panel_b']; lb=ls(aff(b['u']),b['v']); c=s['panel_c']; u=c['u']; z=(x-la[0])@np.linalg.inv(la[1:]);
ops={'raw':x,'cal':z,'b@raw':aff(x)@lb,'b@cal':aff(z)@lb,'binv@cal':(z-lb[0])@np.linalg.inv(lb[1:])}
for k in ('v1','v2'):
 v=c[k]; print(k)
 for name,f in [('square',lambda a:a*a),('cube',lambda a:a**3),('tanh',np.tanh),('sin',np.sin),('abs',np.abs),('rational',lambda a:a/(1+a*a)),('relu',lambda a:np.maximum(a,0))]:
  a=np.c_[aff(u),f(u)];q=ls(a,v); print(name,np.sqrt(np.mean((a@q-v)**2)),np.linalg.norm(q[5:]-(q[1:5])*np.median(q[5:]/q[1:5]))); 
  if name=='square':print(q)
 q=ls(np.c_[aff(u),u**2,u**3],v);print('sq+cub',np.sqrt(np.mean((np.c_[aff(u),u**2,u**3]@q-v)**2)))
 for name,ar in list(ops.items())[:4]: ops[k+'@'+name]=np.c_[aff(ar),ar**2,ar**3]@q
E=lambda k:-np.expm1(-k*t)
fs={'t':lambda p:np.c_[E(p[0]),t], 'twoexp':lambda p:np.c_[E(p[0]),E(p[1])], 'log':lambda p:np.c_[E(p[0]),np.log1p(p[1]*t)], 'pow':lambda p:np.c_[E(p[0]),t**p[1]], 'osc':lambda p:np.c_[E(p[0]),np.sin(p[1]*t)],'relax+drift':lambda p:np.c_[E(p[0]),t*np.exp(-p[1]*t)]}
for k,f in fs.items():
 def res(p):
  a=f(p);return (a@ls(a,T[:52].T)-T[:52].T).ravel()
 q=least_squares(res,[1.5] if k=='t' else [1.6,.35],bounds=(.0001,10));a=f(q.x);print('time',k,q.x,np.std(q.fun),np.mean(abs(a@ls(a,T[52:].T)-T[52:].T)))
 if k=='twoexp': A=a
Y=np.c_[y,ls(A,T.T).T]; print('target stats',Y.min(0),Y.max(0),Y.std(0));
for name,a in ops.items():
 print('coordinates',name,[round(v,3) for v in np.corrcoef(np.c_[Y,a].T)[:3,3:].ravel()]);
 if name.startswith(('v1','v2')):continue
 for k in [0,1,2]:
  best=[]
  for i in range(4):
   r=a[:,i];
   for label,f in [('tanh',np.tanh),('sin',np.sin),('rational',lambda x:x/(1+x*x)),('cube',lambda x:x**3),('sigmoid',lambda x:1/(1+np.exp(-x)))]:
    def feat(p):return np.c_[aff(z),f(p[0]*r+p[1])]
    def err(p):
     a=feat(p)[:52];return a@ls(a,Y[:52,k])-Y[:52,k]
    g=least_squares(err,[1.,0.],bounds=([.1,-2.5],[6,2.5]),max_nfev=100);X=feat(g.x);e=np.mean(abs(X[52:]@ls(X[:52],Y[:52,k])-Y[52:,k]));best.append((round(np.mean(abs(g.fun)),4),round(e,4),i,label,np.round(g.x,3)))
  print('ridge',k,sorted(best,key=lambda p:p[0])[:3])
