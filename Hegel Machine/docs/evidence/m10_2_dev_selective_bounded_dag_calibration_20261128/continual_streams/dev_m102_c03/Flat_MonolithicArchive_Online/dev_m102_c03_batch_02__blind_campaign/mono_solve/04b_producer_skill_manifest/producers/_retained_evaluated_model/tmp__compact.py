import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; T=tr['traj']; y=tr['y']; t=s['eval_times']; N=len(x); off,g=m._calibration(s['panel_a']); z=(x-off)/g; u,v=m._branches(s['panel_c'],z); mm=np.sum(u*v,axis=1); h=u[:,0]*v[:,1]; o=np.ones(N); rng=np.random.default_rng(102); orders=[np.arange(N),np.arange(N)[::-1]]+[rng.permutation(N) for _ in range(24)]; orders += [np.argsort(sign*x[:,j]) for j in range(4) for sign in [1,-1]]
blocks={'o':o,'z':z,'u':u,'v':v,'m':mm,'h':h,'d':u*v,'p':(u[:,:,None]*v[:,None,:]).reshape(N,-1)}
def F(keys): return np.column_stack([blocks[k] for k in keys]);
def fit(f,g,I):
 a,b=f[I],g[I]; pa=np.linalg.pinv(a); pb=np.linalg.pinv(b); Qa=a@pa; Qb=b@pb; P=np.eye(len(I)); ta=T[I]
 def model(theta,ret=False):
  q=-np.expm1(-np.exp(theta)[:,None]*t); aa,bb=q; cross=aa@bb; A=(aa@aa)*P-(cross**2/(bb@bb))*(Qa@Qb); B=Qa@(ta@aa)-(cross/(bb@bb))*Qa@Qb@(ta@bb); ca=np.linalg.solve(A,B); cb=Qb@(ta@bb-cross*ca)/(bb@bb);
  if ret:return (f@(pa@ca))[:,None]*aa+(g@(pb@cb))[:,None]*bb, np.exp(theta),pa@ca,pb@cb
  return (ca[:,None]*aa+cb[:,None]*bb-ta).ravel()
 sol=least_squares(model,np.log([.4,1.8]),bounds=(np.log([.12,1.0]),np.log([.9,3.5])),max_nfev=60); return model(sol.x,True)
results=[]
for k,l in [('ovh','ozum'),('ovh','ozumh'),('ovh','ozumhv'),('ovh','oum'),('ovh','oumh'),('ovh','ozm'),('ovh','ozmh'),('ouvh','ozum'),('ouvh','ozumh'),('ozuvh','ozumh'),('ovh','ozuvd'),('ovh','ozuvp'),('ozuvp','ozuvp')]:
 E=[]
 for order in orders:
  I,J=order[:52],order[52:]; P,r,a,b=fit(F(k),F(l),I); err=abs(P[J]-T[J]); E.append([err.mean(),err.mean(0).max(),err.mean(1).max()]);
  if len(E)==1: first=(r,np.sqrt(np.mean((P[I]-T[I])**2)),a,b)
 E=np.array(E); results.append((E.max(0)[1],k,l,E[0],E[:26].max(0),E[26:].max(0),first))
for result in sorted(results,key=lambda r:r[0]): print(result)
for k in ['oum','ozum','ozuvm','ovm','om','ouvm']:
 E=[]
 for order in orders:
  I,J=order[:52],order[52:]; f=F(k); e=(f@np.linalg.lstsq(f[I],y[I],rcond=None)[0]-y); E.append(np.mean(abs(e[J])))
 print('scalar',k,E[0],max(E[:26]),max(E[26:]))
