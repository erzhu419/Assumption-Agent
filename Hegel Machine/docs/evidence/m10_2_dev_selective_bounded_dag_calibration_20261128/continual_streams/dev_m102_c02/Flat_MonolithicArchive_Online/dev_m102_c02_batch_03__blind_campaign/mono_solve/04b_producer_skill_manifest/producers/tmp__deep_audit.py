exec(open('/tmp/audit_solver.py').read().split('results=[]')[0])
from itertools import combinations
from scipy.optimize import minimize_scalar
panel=s['panel_a']; a,b,c=cal; u,v=panel['u'],panel['v']; nxt=np.roll(u,-1,axis=1); delta=np.linalg.norm(a+b*u+c[0]*nxt+c[1]*u*nxt+c[2]*u**3-v,axis=1); good=delta<.001
print('Panel A gross rows, inlier maximum',np.where(~good)[0],delta[good].max()); print('Calibration jackknife',max(np.max(np.abs(np.r_[*m._calibration({k:d[np.arange(len(u))!=j] for k,d in panel.items()})]-np.r_[*cal])) for j in range(len(u))))
print('Study keys',list(s),'train keys',list(t),'times',s['eval_times']); print('Panel B affine RMSE',np.mean((m._affine(bp['u'])@m._fit(m._affine(bp['u']),bp['v'])-bp['v'])**2)**.5)
p=w[:,0]*w[:,2]; q=w[:,1]*w[:,3]; y=t['y']; D=m._affine(w); unshared=np.c_[D,p,q]; shared=np.c_[D,p+q]; origin=shared[:,1:]; full=m._quadratic(z)
rat=[]
for i,j in combinations(range(4),2):
 for c0 in (-1,1):
  pair=(w[:,i]+c0*w[:,j])**2
  for a0 in (.02,.08,.22,.65): rat.append(shared/(1+a0*pair[:,None]))
res=[]
for name,te in splits:
 tr=np.setdiff1d(idx,te)
 def fit_err(h): return np.mean(abs(h[te]@m._fit(h[tr],y[tr])-y[te]))
 def press(h):
  u=np.linalg.qr(h[tr],mode='reduced')[0]; return np.mean(abs((y[tr]-u@(u.T@y[tr]))/(1-(u*u).sum(1))))
 rational=min(rat,key=press)
 tm=s['eval_times']; traj=t['traj']; aa=m._affine(z)
 def basis(k):return -np.expm1(-np.outer(k,tm))
 beta=m._fit(aa[tr],traj[tr]); best=None
 for k0 in np.geomspace(.07,5,40):
  for k1 in np.geomspace(.08,9,42):
   if k1<=1.15*k0: continue
   bb=basis([k0,k1]); coef=m._fit(bb.T,beta.T).T
   loss=np.mean((aa[tr]@(coef@bb-beta))**2)
   if best is None or loss<best[0]: best=(loss,coef@bb,k0,k1)
 state=aa[te]@best[1]; errors=[fit_err(h) for h in (full,unshared,shared,origin,rational)]+[np.mean(abs(state-traj[te])),np.max(np.mean(abs(state-traj[te]),axis=0))]; res.append(errors)
 if name=='52/20': print('52/20 columns full/unshared/shared/origin/rational/exp-traj/exp-worst-time',errors,'rates',best[-2:])
print('median',np.median(res,axis=0),'worst',np.max(res,axis=0))
