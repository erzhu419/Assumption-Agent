import runpy,contextlib,io,numpy as np
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','di','dt','t','tau','theta','bridge','v','vr')})
x=m._features(di); past,hold,_=m._coupon(di,x,bridge); labs=di['lab_id']; raw=di['coupon_features_raw']; f=theta[2]+x@theta[3:]; H=-np.expm1(-t[:,None]/tau)@theta[:2]; Y=np.c_[v,vr]-hold[:,None]*np.r_[H,np.sum(theta[:2]/tau)]; a=np.array([.322898,1.496942]); c=np.array([.060623,.962508]); B=np.r_[-np.expm1(-t[:,None]/a)@c+.16156*t,np.sum(c/a)+.16156]; T=np.r_[t,1.]; amp=((Y+.001001*T)@B)/(B@B); e=amp-f
print('lab missing score and residual per coupon:'); print(np.c_[labs,np.isnan(raw).any(1),f,amp,e].round(6)); print('correlation score-residual',np.corrcoef(np.c_[f,f**2,f**3,hold,past, x, e].T)[-1,:-1].round(3))
for label,F in [('linear',f[:,None]),('free',x),('quadratic',np.c_[f,f**2]),('absolute',np.c_[f,abs(f)]),('cubic',np.c_[f,f**3]),('rational',np.c_[f, f/(1+f*f)]),('missing',np.c_[f,np.isnan(raw).astype(float)])]:
 A=np.c_[np.ones(len(f)),F]; errs=[]
 for lab in np.unique(labs):
  sel=labs!=lab; coef=np.linalg.lstsq(A[sel],amp[sel],rcond=None)[0]; E=(A[~sel]@coef)[:,None]*B-.001001*T-Y[~sel]; errs.append(np.sqrt(np.mean(E[:,[0,-2,-1]]**2,axis=0)))
 print(label,'worst lab',np.max(errs,axis=0).round(6))
