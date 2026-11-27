exec(open('/tmp/rheo_diag.py').read().split("for k,z,xx")[0])
from scipy.optimize import least_squares
p,h=m._loads(q,b); y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; dr=np.c_[np.ones(len(q['lab_id'])),xq]@th[2:];
def basis(ta): return np.r_[np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1][None,:]]
scale=np.r_[np.full(len(t),.004),.004];
def amps(ta):
 M=basis(ta); a=np.linalg.lstsq(M/scale[:,None],(y/scale).T,rcond=None)[0].T;return a,a@M.T-y
fit=least_squares(lambda z:(amps(np.exp(z))[1]/scale).ravel(),np.log(tau),bounds=(np.log([.05,1]),np.log([1,25]))); A,E=amps(tau); print('free chamber taus, projected RMSE fixed/free:',np.exp(fit.x),np.sqrt(np.mean(E**2)),np.sqrt(np.mean(amps(np.exp(fit.x))[1]**2))); print('projection singular values',np.linalg.svd(basis(tau),compute_uv=False))
one=np.ones(len(h)); mem=h[:,None]-m._design(xq,p,h,dt,t,tau)[1][:,:2]*tau
print('amplitudes minus theta*hold, drift, sum of history:',np.c_[A[:,:2]-h[:,None]*th[:2],A[:,2],dr,p.sum(1),p[:,0],p[:,-1]].round(4));
for k in range(3):
 tar=A[:,k]-(h*th[k] if k<2 else 0);print('AMPLITUDE',k)
 for name,Q in [('offset',one),('drift',np.c_[one,dr]),('drift+hold',np.c_[one,dr,h]),('history',np.c_[one,p]),('memory',np.c_[one,mem]),('memory+drift',np.c_[one,mem,dr]),('mem+hold+drift',np.c_[one,mem,h,dr]),('history+drift+hold',np.c_[one,p,dr,h]),('features+memory+hold',np.c_[one,xq,mem,h]),('features',np.c_[one,xq])]:
  if Q.ndim==1: Q=Q[:,None]
  coef=np.linalg.lstsq(Q,tar,rcond=None)[0]; er=tar-Q@coef; folds=[]
  for g in np.unique(q['lab_id']):
   s=q['lab_id']==g; folds.append(np.sqrt(np.mean((tar[s]-Q[s]@np.linalg.lstsq(Q[~s],tar[~s],rcond=None)[0])**2)))
  print(name,'fit/worstLab',round(np.sqrt(np.mean(er**2)),5),round(max(folds),5),'coef',coef.round(4))
print('diag missing rows',np.nonzero(np.isnan(q['coupon_features_raw']).any(1))[0]); print('shapes/history correlations',p[:5].round(3))
