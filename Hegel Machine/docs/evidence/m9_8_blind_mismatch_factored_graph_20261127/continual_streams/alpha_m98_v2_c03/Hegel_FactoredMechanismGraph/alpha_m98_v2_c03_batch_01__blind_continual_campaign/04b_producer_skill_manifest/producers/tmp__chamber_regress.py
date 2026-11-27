exec(open('/tmp/explore_chamber.py').read().split('for tt in [taus]:')[0])
from scipy.optimize import least_squares
from itertools import product
h=cq[1]; P=cq[0]; X=xq.copy(); raw=q['coupon_features_raw']; F=np.c_[np.ones(len(X)),X]; n=len(h)
lab=np.r_[tr['lab_id'],d['test']['lab_id']]; strata=np.r_[tr['stratum'],d['test']['stratum']]; _, inv, ct=np.unique(np.c_[lab,strata],axis=0,return_inverse=True,return_counts=True); w=1/ct[inv]; joint=np.vstack([tr['coupon_features_raw'],d['test']['coupon_features_raw']]); fill=np.average(x,axis=0,weights=w); X=np.where(np.isnan(raw),fill,X); F=np.c_[np.ones(n),X]; dr=F@th[2:]; z=h[:,None]-m._design(cq,dt,np.array([1.]),taus)[:,-1,:2]*taus
B=basis(taus); amps=np.linalg.lstsq(B[:,:2],y.T,rcond=None)[0].T
print('balanced fill',fill,'diagnostic missing rows',np.where(np.isnan(raw))[0]); print('h, z0,z1, drift, amp0,amp1, delta0,delta1'); print(np.round(np.c_[h,z,dr,amps,amps-h[:,None]*th[:2]],5))
one=np.ones(n)
models={'hold':h[:,None],'hold+drift':np.c_[h,dr], 'hold+offset+drift':np.c_[h,one,dr],'hold+features':np.c_[h,F],'hold+features+history':np.c_[h,F,z], 'hold+drift+history':np.c_[h,dr,z],'hold+drift_polynomial':np.c_[h,one,dr,dr**2,dr**3], 'hold+drift_poly_interaction':np.c_[h,one,dr,dr**2,h*dr], 'hold+absdrift':np.c_[h,one,dr,abs(dr)],'hold+lab+drift':np.c_[h,dr,np.eye(4)[q['lab_id']]], 'hold+stratum+drift':np.c_[h,dr,np.eye(3)[q['stratum']]]}
for key,V in models.items():
 def run(i):
  k=np.linalg.lstsq(V[i],amps[i],rcond=None)[0]; return V@k@B[:,:2].T,k
 p,k=run(np.arange(n)); e=np.array([run(np.arange(n)!=i)[0][i]-y[i] for i in range(n)]); print(key,'fit',stat(p-y),'LOO',stat(e),'worst',np.round([max(np.sqrt(np.mean(e[:,:-1]**2,axis=1))),max(abs(e[:,-1]))],5),'coef',np.round(k.T,4).tolist())
print('centered amps against centered feature values'); sel=np.isfinite(raw).all(1); print(np.corrcoef(np.c_[X,h,z,dr,amps-h[:,None]*th[:2]][sel].T)[-2:])
