exec(open('/tmp/chamber_probe.py').read().split('for name,d in')[0])
from scipy.optimize import least_squares
Y=dg['strain_curves_obs']-dg['strain_0'][:,None]; R=dg['instant_rate_obs']; y=np.c_[Y,R]; N=len(Y); lab=dg['lab_id']; ix=np.arange(N)
p,h,d=m._coupon_inputs(dg,m._features(tr,dg)[n:],b)
def basis(q): return np.r_[-np.expm1(-t[:,None]/q),(1/q)[None,:]]
def prof(q):
 B=basis(q); return (np.linalg.lstsq(B,y.T,rcond=None)[0].T@B.T-y).ravel()
f=least_squares(prof,taus,bounds=(.05,25)); q=f.x; B=basis(q); a=np.linalg.lstsq(B,y.T,rcond=None)[0].T
print('zero-drift unrestricted taus',q,'curve/rate rms',np.sqrt(np.mean(prof(q).reshape(N,-1)**2,axis=0)).round(5)); print('per-coupon amplitude in stress units minus hold, alongside hold and history:'); print(np.c_[a/th[:2]-h[:,None],h,p].round(4))
rawp=dg['past_channel_raw']; rawh=dg['hold_channel_raw']; loads_p=rawp@b[1:]+b[0]; loads_h=rawh@b[1:]+b[0]; after=np.cumsum(dt[::-1])[::-1]-dt; before=np.cumsum(dt)-dt
z=lambda X,s,lag=after: np.einsum('ns...,sb->nb...',X,-np.expm1(-dt[:,None]/s)*np.exp(-lag[:,None]/s))
mem=z(p,q); rev=z(p,q,before)
print('branch coefficient y minus benchtop history gap vs stress/past features, Pearson:')
names=['hold','baseline_fast_mem','baseline_slow_mem','reverse_fast_mem','reverse_slow_mem','past_mean','past_min','past_max','past_abs_mean','past_sq_mean']+[f'past{k}' for k in range(len(dt))]+[f'hold_channel{k}' for k in range(4)]
M=np.c_[h,mem,rev,p.mean(1),p.min(1),p.max(1),abs(p).mean(1),(p*p).mean(1),p,loads_h]; co=np.corrcoef(np.c_[a/th[:2]-h[:,None],M].T)[:2,2:]; print(sorted(zip(names,*co.round(3)),key=lambda row:-max(abs(row[1]),abs(row[2]))))
for name,g in {'hold only':[h,h], 'baseline gap':[h-mem[:,0],h-mem[:,1]],'reverse gap':[h-rev[:,0],h-rev[:,1]],'plus memory':[h+mem[:,0],h+mem[:,1]],'history independent':[np.c_[h,p],np.c_[h,p]],'clipped stress':[np.c_[h,z(np.maximum(p,0),q)[:,0]],np.c_[h,z(np.maximum(p,0),q)[:,1]]],'weighted nonlinear stress':[np.c_[h,mem[:,0],z(p*p,q)[:,0]],np.c_[h,mem[:,1],z(p*p,q)[:,1]]],'separate hold channels':[loads_h,loads_h],'hold and same mode channel memories':[np.c_[loads_h,z(loads_p,q)[:,0]],np.c_[loads_h,z(loads_p,q)[:,1]]],'hold and powered hold':[np.c_[h,h*h,np.ones(N)],np.c_[h,h*h,np.ones(N)]]}.items():
 g=[np.reshape(v,(N,-1)) for v in g]; v=np.concatenate([g[k][:,None,:]*B[None,:,k,None] for k in range(2)],axis=-1)
 def run(j):
  return np.linalg.lstsq(v[j].reshape(len(j)*y.shape[1],-1),y[j].ravel(),rcond=None)[0]
 folds=[]
 for k in np.unique(lab):
  mask=lab==k; e=v[mask]@run(ix[~mask])-y[mask]; folds.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]);
 print(name,'worst LOLO',np.max(folds,0).round(5),'coeff',run(ix).round(4))
