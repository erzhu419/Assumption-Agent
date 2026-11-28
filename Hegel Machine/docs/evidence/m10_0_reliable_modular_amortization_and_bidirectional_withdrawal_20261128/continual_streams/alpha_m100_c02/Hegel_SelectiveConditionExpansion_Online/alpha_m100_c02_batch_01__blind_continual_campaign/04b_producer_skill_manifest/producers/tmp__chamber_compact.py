exec(open('/tmp/chamber_law.py').read().split('for i,x in enumerate(xs):')[0])
x=xs[1]; drift=x@th[2:]; anchor=hold[:,None]*th[:2]; E=y-anchor@F[:,:2].T
# Unit initial slope in all three time coordinates.
T=F*np.r_[tau,1];
print('observed chamber drift vs coupon drift',np.c_[E[:,-1],drift,E[:,-1]-drift]);
print('relative rate weights matrix',np.linalg.lstsq(drift[:,None,None]*T[None,:,:] if False else (drift[:,None,None]*T[None,:,:]).reshape(-1,3),E.ravel(),rcond=None)[0]);
rate_sig=E[:,-1]; C=np.linalg.lstsq((rate_sig[:,None,None]*(T[:,:2]-T[:,2:])).reshape(-1,2),(E-rate_sig[:,None]*T[:,2]).ravel(),rcond=None)[0]; print('temporal allocation using measured rate',C,1-C.sum()); print('shape RMSE using observed rate', np.sqrt(np.mean((E-rate_sig[:,None]*(T@np.r_[C,1-C.sum()]))[:,:-1]**2)))
for scale in [0,0.5,1]:
 print('custom reference mix',scale)
 gg=dict(g); a=np.vstack([g['coupon_features_raw']]).copy(); ref=g['ref_coupons']; labs=g['lab_id']; xf=np.empty_like(a); labs0=tr['lab_id']; labs1=te['lab_id'];
 for lab in np.unique(labs):
  rows=labs==lab; panel=ref[rows]; p0=np.concatenate([tr['ref_coupons'][labs0==lab],te['ref_coupons'][labs1==lab]]); center=lambda p:np.median(p.mean(0)); std=lambda p:np.median(p.std(0)); xf[rows]=(a[rows]-((1-scale)*center(panel)+scale*center(p0)))/((1-scale)*std(panel)+scale*std(p0))
 _,cell,n=np.unique(np.c_[labs,g['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[cell].reshape(-1,1); present=~np.isnan(xf); xf=np.where(present,xf,np.nansum(w*xf,axis=0)/(w*present).sum(0)); dr=th[2]+xf@th[3:]; res=E-dr[:,None]*(T@np.r_[C,1-C.sum()]);print('strain/rate RMSE and max rate',np.sqrt(np.mean(res[:,:-1]**2)),np.sqrt(np.mean(res[:,-1]**2)),max(abs(res[:,-1])));
