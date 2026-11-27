exec(open('/tmp/joint.py').read().split("for kind in ('standard'")[0])
# The verification coupons form their own panel. Only their design variables
# are pooled across folds; no held-out strain or rate enters the fit.
zz=di['coupon_features_raw'].copy()
for lab in np.unique(labs):
    rows=labs==lab
    panel=di['ref_coupons'][rows]
    zz[rows]=(zz[rows]-np.median(panel.mean(0)))/np.median(panel.std(0))
missing=np.isnan(zz).any(1)
zz=np.where(np.isnan(zz),np.nanmean(zz,0),zz)
drift=th[2]+zz@th[3:]
lean=np.c_[hd,drift]
full=np.c_[np.ones(N),lean]

def report(e):
    rms=lambda a: np.sqrt(np.mean(a*a))
    return np.round([rms(e[:,:-1]),rms(e[:,-1]),max(rms(e[labs==l,:-1]) for l in np.unique(labs)),max(rms(e[labs==l,-1]) for l in np.unique(labs)),np.max(np.sqrt(np.mean(e[:,:-1]**2,0))),rms(e[missing,:-1]),rms(e[missing,-1])],6)
for name, X in [('zero-intercept',lean),('intercept',full)]:
    fit,coef,u=run([X]*3,[])
    oo=np.empty_like(y)
    for lab in np.unique(labs):
        ix=np.flatnonzero(labs==lab)
        oo[ix]=(run([X]*3,ix)[0]-y)[ix]
    print(name,'fit',report(fit-y),'LOLO',report(oo),'chamber taus',u,'coefficients',coef.reshape(3,-1),flush=True)
    if name=='zero-intercept':
        with open('/tmp/chamber_constants.py','w') as f:
            f.write('\n# Identified from chamber verification coupons, separate from the\n# intrinsic material fit on benchtop observations.\n')
            f.write('_CHAMBER_TIMES = np.array('+repr(u.tolist())+')\n')
            f.write('_CHAMBER_COEFFICIENTS = np.array('+repr(coef.reshape(3,2).tolist())+')\n')

print('Benchtop panel check: [strain median absolute residual, rate RMSE]')
for name, features in [('joint prescribed',x[:n]),('train panel only',m._features(tr,tr)[:n])]:
    a,b=m._design(features,*m._loads(tr,br),dt,t,tau)
    ec=(a@th).reshape(n,-1)+tr['strain_0'][:,None]-tr['strain_curves_obs']
    er=b@th-tr['instant_rate_obs']
    print(name,np.median(abs(ec)),np.sqrt(np.mean(er**2)))
