import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; np.set_printoptions(precision=5,suppress=True,linewidth=130)
def aff(p,key,rob=False):
 u=p['u']; v=p[key]; F=np.c_[np.ones(len(u)),u]; ids=np.arange(len(u));
 for _ in range(7 if rob else 1):
  c=np.linalg.lstsq(F[ids],v[ids],rcond=None)[0]; e=np.linalg.norm(F@c-v,axis=1)
  if rob: ids=np.where(e<max(.001,4*np.median(e)))[0]
 return lambda q:q@c[1:]+c[0]
a=aff(s['panel_a'],'v',True); b=aff(s['panel_b'],'v'); c=aff(s['panel_c'],'v1'); d=aff(s['panel_c'],'v2')
def basis(p):
 k,dam,w=p; return np.c_[-np.expm1(-k*t), np.exp(-dam*t)*np.sin(w*t)]
def amp(z,p): return np.linalg.lstsq(basis(p),z.T,rcond=None)[0].T
def temp(z): return least_squares(lambda p:(amp(z,p)@basis(p).T-z).ravel(),[.5,.2,1.1],bounds=([.01,-.5,.3],[5,2,3])).x
p=temp(z[:56]); v=amp(z,p); print('temporal par',p,'heldout reconstruction MAE',np.abs(v[56:]@basis(p).T-z[56:]).mean(), 'amp std',v.std(0))
def poly(q,k): return np.c_[np.ones(len(q)),*[np.prod(q[:,ii],axis=1) for j in range(1,k+1) for ii in combinations_with_replacement(range(q.shape[1]),j)]]
def lr(f,h,ids=np.arange(56)): return np.linalg.lstsq(f[ids],h[ids],rcond=None)[0]
w=np.c_[y,v];
for k in (1,2,3,4):
 f=poly(a(x),k); cc=lr(f,w); e=np.abs(f@cc-w); print('poly',k,'fit',e[:56].mean(0),'test',e[56:].mean(0));
 if k<=2: print('coef',cc)
for name,q in [('A',a(x)),('B',b(x)),('BA',b(a(x))),('AB',a(b(x))),('C',c(a(x))),('D',d(a(x))),('amp',v),('Aamp',np.c_[a(x),v])]:
 print(name,'scores')
 for nm,f in [('tanh',poly(np.tanh(q),2)),('sin',poly(np.sin(q),2)),('tanh+q',np.c_[poly(q,2),np.tanh(q)]),('sin+q',np.c_[poly(q,2),np.sin(q)])]:
  e=np.abs(f@lr(f,w)-w); print(nm,e[56:].mean(0),end=' ')
 print()
print('amp ranges',v.min(0),v.max(0)); print('corr y,amps',np.corrcoef(w.T)); print('amp vs x rows',np.c_[x,v,y][:8]);
