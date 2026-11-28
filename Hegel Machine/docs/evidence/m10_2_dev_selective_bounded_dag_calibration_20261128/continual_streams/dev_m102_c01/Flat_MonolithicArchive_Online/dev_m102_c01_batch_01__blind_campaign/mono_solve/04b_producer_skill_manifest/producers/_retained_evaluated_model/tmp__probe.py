import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive'); import dev_m102_c01_batch_00_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=(tr[k] for k in ('x','y','traj')); sv=np.linalg.svd(t,full_matrices=False)[2]; tt=t@sv[0]; Y=np.c_[y,tt]; np.set_printoptions(precision=5,suppress=True)
a=s['panel_a']; sl,off=old._calibration(a); print('t',s['eval_times'],'telemetry',s['telemetry_vec'],'a',sl,off,'diag resid',np.sort(np.linalg.norm(a['v']-a['u']*sl-off,axis=1))); b=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(b['u'])),b['u']],b['v'],rcond=None)[0]; print('B',B)
bmap=lambda z: B[0]+z@B[1:]; c=s['panel_c']; u=c['u']; ops={};
for key in ('v1','v2'):
 for name,fun,br in [('tanh',lambda z,k:np.tanh(k*z),(.05,5)),('sin',lambda z,k:np.sin(k*z),(.05,5)),('rat',lambda z,k:np.c_[z,z*z/(1+k*z*z)],(.01,5)),('quad',lambda z,k:np.c_[z,z*z],(.01,5))]:
  def fit(k):
   F=np.c_[np.ones(len(u)),fun(u,k)]; W=np.linalg.lstsq(F,c[key],rcond=None)[0]; return W,np.sqrt(np.mean((F@W-c[key])**2))
  k=minimize_scalar(lambda k:fit(k)[1],bounds=br,method='bounded').x; W,r=fit(k); print(key,name,k,r); ops[key,name]=(r,lambda z,k=k,W=W,fun=fun:np.c_[np.ones(len(z)),fun(z,k)]@W)
Q={}; maps={'B':bmap,'Bi':lambda z:(z-B[0])@np.linalg.inv(B[1:])}; maps.update({key:min([v for (j,_),v in ops.items() if j==key],key=lambda v:v[0])[1] for key in ('v1','v2')})
for key,z in [('x',x),('A-',(x-off)/sl),('A+',x*sl+off)]:
 Q[key]=z
 for k,m in maps.items():
  Q[key+k]=m(z)
  for k2 in ('v1','v2'):
   if k not in ('v1','v2'):Q[key+k+k2]=maps[k2](m(z))
def p(z,d):return np.c_[np.ones(len(z)),*[np.prod(z[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(4),k)]]
rows=[]; rng=np.random.default_rng(91); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(8)]
for key,z in Q.items():
 for label,F in [(f'p{d}',p(z,d)) for d in (1,2,3)]+[(f'p2+{fn.__name__}{k}',np.c_[p(z,2),fn(k*z)]) for fn in (np.sin,np.tanh) for k in (1.,2.,3.)]+[(f'p2+tanhprod{k}',np.c_[p(z,1),np.tanh(k*(p(z,2)[:,5:]))]) for k in (1.,2.)]:
  err=[]
  for ix in splits:
   a,b=ix[:52],ix[52:]; W=np.linalg.lstsq(F[a],Y[a],rcond=None)[0]; err.append(np.mean(abs(Y[b]-F[b]@W),axis=0))
  rows.append((key,label,err[0],np.mean(err,axis=0),np.max(err,axis=0)))
for j in (0,1):
 print('rank',j); print(*sorted(rows,key=lambda r:r[3][j])[:16],sep='\n')
