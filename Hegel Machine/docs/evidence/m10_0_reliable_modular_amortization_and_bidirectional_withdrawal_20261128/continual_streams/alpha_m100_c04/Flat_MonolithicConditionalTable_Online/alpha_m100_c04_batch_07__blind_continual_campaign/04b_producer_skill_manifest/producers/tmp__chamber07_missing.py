exec(open('/tmp/chamber07_law.py').read().split('blocks=')[0])
raw=z['coupon_features_raw']; complete=np.isfinite(raw).all(1); xx=m._features(z,z)[len(h):]; drift=theta[2]+xx@theta[3:]; F=np.column_stack((A,t.tolist()+[1.])); R=Y-base
print('diag complete counts by lab',[(int(j),int(sum(complete&(labs==j))),int(sum(labs==j))) for j in np.unique(labs)]); print('test missing',np.isnan(te['coupon_features_raw']).sum(0)); print('times',t,'dt',dt)
for s in (complete,np.ones(len(h),bool)):
 q=drift[s,None,None]*F[None,:,:]; c=np.linalg.lstsq(q.reshape(-1,3),R[s].ravel(),rcond=None)[0]; o=R-drift[:,None]*(F@c); print('coeff',c,'complete/missing rms',[(np.sqrt(np.mean(o[e,:-1]**2)),np.sqrt(np.mean(o[e,-1]**2))) for e in (complete,~complete)]); print('per-coupon residual and inferred drift adjustment',np.c_[labs,~complete,np.sqrt(np.mean(o**2,axis=1)),o@(F@c)/np.linalg.norm(F@c)**2].round(5).tolist());
for p in (tr,te,z):
 print('ref unique panel sizes',[(int(j),len(np.unique(p['ref_coupons'][p['lab_id']==j],axis=0)),np.median(p['ref_coupons'][p['lab_id']==j].mean(0)),np.median(p['ref_coupons'][p['lab_id']==j].std(0))) for j in np.unique(p['lab_id'])]);
print('feature routine:'); import inspect; print(inspect.getsource(m._features));
