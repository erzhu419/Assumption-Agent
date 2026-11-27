exec(open('/tmp/chamber_probe.py').read().split('def project')[0])
# Use the same reference alignment for fitting and deployment, including diagnostic covariates only in the projection below.
def fixed_features(a,b,c):
    raw=np.vstack([a['coupon_features_raw'],b['coupon_features_raw']]); ref=np.vstack([a['ref_coupons'],b['ref_coupons']]); labs=np.r_[a['lab_id'],b['lab_id']]; strata=np.r_[a['stratum'],b['stratum']]; zz=c['coupon_features_raw'].copy()
    for l in np.unique(labs):
        k=labs==l; shift=np.median(ref[k].mean(0)); scale=np.median(ref[k].std(0)); raw[k]=(raw[k]-shift)/scale; z=c['lab_id']==l; zz[z]=(zz[z]-shift)/scale
    _,inv,n=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); w=1/n[inv]; center=np.nansum(raw*w[:,None],axis=0)/(np.isfinite(raw).T @ w)
    return np.where(np.isfinite(zz),zz,center)
Q=fixed_features(a,b,c)
age=np.cumsum(dt[::-1])[::-1]-dt
Z=P@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); G=H[:,None]-Z
D=theta[2]+Q@theta[3:]; U=P[:,-1]; V=H-U
F=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.])); obs=np.c_[Y,R]; C=np.linalg.lstsq(F,obs.T,rcond=None)[0].T
print('fixed tau per-coupon projection noise',np.sqrt(np.mean((C@F.T-obs)**2,axis=0)).round(5))
print('columns coef0, coef1, drift, H, G0, G1, D, last'); print(np.c_[C,H,G,D,U].round(4))
zsets={'H':H[:,None], 'H,G0,G1':np.c_[H,G], 'G0,G1,D':np.c_[G,D], 'H,G0,G1,D':np.c_[H,G,D], 'H,G0,G1, last,D':np.c_[H,G,U,D], 'H,G0,G1,X':np.c_[H,G,Q], 'H, full history, D':np.c_[H,P,D], 'H, full history, X':np.c_[H,P,Q], 'H, hold channels, G0,G1,D':np.c_[c['hold_channel_raw'],G,D], 'H,G0,G1,D,quadratic':np.c_[H,G,D,H**2,H[:,None]*G,G**2,G[:,0]*G[:,1],H*D,D**2], 'H,G0,G1,D,V|V|':np.c_[H,G,D,V*abs(V)], 'H,G0,G1,D,rational':np.c_[H,G,D,G/(1+G**2),G/(1+H[:,None]**2)], 'H,G0,G1,D,feature coupling':np.c_[H,G,D,D[:,None]*G,H*D]}
for k,W in zsets.items():
    W=np.c_[np.ones(len(W)),W]; coeff=np.linalg.lstsq(W,C,rcond=None)[0]; print(k,'coefs',coeff.round(4).tolist() if W.shape[1]<7 else '')
    E=np.empty_like(obs)
    for l in np.unique(lab):
        s=lab!=l; E[~s]=W[~s]@np.linalg.lstsq(W[s],obs[s],rcond=None)[0]-obs[~s]
    err=W@coeff@F.T-obs
    def rms(e):return np.round(np.sqrt(np.mean(e**2,axis=0)),4)
    print('  fit',rms(np.c_[err[:,:-1].mean(1),err[:,-1]]),'LOLO',rms(np.c_[E[:,:-1].mean(1),E[:,-1]]),'worst',np.round(np.max(np.abs(E)),4))
for j in range(3):
    print('coef',j,'correlations with hold, gaps, drift, history, raw features',np.corrcoef(np.c_[C[:,j],H,G,D,P,Q].T)[0,1:].round(3))
