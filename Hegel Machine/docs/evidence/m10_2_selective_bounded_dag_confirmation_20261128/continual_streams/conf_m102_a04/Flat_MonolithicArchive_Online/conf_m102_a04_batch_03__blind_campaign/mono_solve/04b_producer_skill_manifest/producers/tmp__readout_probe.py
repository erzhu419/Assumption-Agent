import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; z=b._calibrate(s['panel_a'],d['x']); g,h=np.split(b._panel_maps(s['panel_c'],z),2,1); y=d['y']; T=d['traj']; t=s['eval_times']; N=len(y); prod=g*h; dot=prod.sum(1); cross=(g[:,:,None]*h[:,None,:]).reshape(N,16); base=b.one(np.c_[g,h]); F=b._time_basis(t,T); amps=b.lin(F,T.T).T
for name,A in [('dot',np.c_[base,dot]),('diag',np.c_[base,prod]),('cross',np.c_[base,cross])]:
 co=b.lin(A,np.c_[y,amps]); print(name,'coefficients',np.round(co,5).T,'RMSE',np.std(A@co-np.c_[y,amps],axis=0))
rng=np.random.default_rng(54312); order=np.argsort(np.max(abs(z),axis=1)); perms=[np.arange(N),order,order[::-1]]+[rng.permutation(N) for _ in range(40)]
reg=[0,.0001,.001,.01]; errors={}
def add(key,err):errors.setdefault(key,[]).append(err)
sig={'dot':dot,'g2':(g*g).sum(1),'h2':(h*h).sum(1),'z2':(z*z).sum(1),'p2':(prod*prod).sum(1),'sum2':((g+h)**2).sum(1),'diff2':((g-h)**2).sum(1),'dot2':dot**2}
for p in perms:
 a,q=p[:52],p[52:]; f=b._time_basis(t,T[a]); amp=b.lin(f,T[a].T).T
 for rr in reg:
  for name,M in [('dot',np.c_[g,h,dot]),('diag',np.c_[g,h,prod]),('cross',np.c_[g,h,cross])]:
   out=b._readout(M[p],np.c_[y[a],amp],52,rr); add((name,rr,'y'),abs(out[:,0]-y[q]).mean()); add((name,rr,'T'),abs(out[:,1:]@f.T-T[q]).mean())
 for name,v in sig.items():
  for mode in ['numer','all','tanh','sin','extra']:
   def design(k):
    if mode=='numer':return np.c_[base,dot/(1+k*v)]
    if mode=='all':return np.c_[base,dot]/(1+k*v[:,None])
    if mode=='tanh':return np.c_[base,np.tanh(k*dot)/k]
    if mode=='sin':return np.c_[base,np.sin(k*dot)/k]
    return np.c_[base,dot,v]
   def resid(pp):
    A=design(pp[0])[a]; return A@b.lin(A,y[a])-y[a]
   lo=max(-.7/max(v.max(),1e-8),-.8); hi=min(-.7/min(v.min(),-1e-8),3.)
   if mode in ('tanh','sin'): lo,hi=.0001,3.
   k=least_squares(resid,[min(.22,hi/2)],bounds=(lo,hi)).x[0] if mode!='extra' else 0.
   A=design(k); add((name,mode,'y'),abs(A[q]@b.lin(A[a],y[a])-y[q]).mean())
   if p is perms[0] and name=='dot': print('scalar fit',mode,k,np.std(resid([k])))
for idx in ['y','T']:
 for key,v in sorted([(k,v) for k,v in errors.items() if k[-1]==idx],key=lambda a: max(a[1])+.6*np.mean(a[1]))[:16]: print(key,'gate',round(v[0],6),'edge',np.round(v[1:3],6),'mean/max',np.round([np.mean(v),max(v)],6))
