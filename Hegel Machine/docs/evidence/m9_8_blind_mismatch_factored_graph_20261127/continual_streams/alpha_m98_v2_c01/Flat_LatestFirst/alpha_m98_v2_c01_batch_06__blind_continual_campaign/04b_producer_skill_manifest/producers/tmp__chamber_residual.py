exec(open('/tmp/corrected_probe.py').read().split("print('diagnostic reference")[0])
np.set_printoptions(precision=5,suppress=True)
_,xs=m._features(g,g); L=np.log1p(t); E=-np.expm1(-t[:,None]/tau)
G=y-(h[:,None]*th[:2])@E.T; dr_obs=(G@L)/(L@L); dr_rate=r-h@(np.ones((1,2))) if False else r-h*np.sum(th[:2]/tau)
raw=g['coupon_features_raw']; print('diag columns: lab stratum missing-mask drift_curve drift_rate drift_diag drift_joint curve_residual_max'); print(np.c_[g['lab_id'],g['stratum'],~np.isfinite(raw),dr_obs,dr_rate,th[2]+xs@th[3:],dr,np.max(abs(G-dr_obs[:,None]*L),axis=1)])
print('lab/strata missing counts train,test,diagnostics',[(len(z['lab_id']),np.isnan(z['coupon_features_raw']).sum(0)) for z in (tr,te,g)])
for label,xt_ in [('joint',xt),('train only',m._features(tr,tr)[0])]:
 st=spec(tr,xt_); ta,we=m._fit(st,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); cc,rrr=m._branches(st,dt,t,ta); drt=st[2]@we[2:]; ec=cc@we[:2]+drt[:,None]*t+tr['strain_0'][:,None]-tr['strain_curves_obs']; er=rrr@we[:2]+drt-tr['instant_rate_obs']; print(label,'tau, weights',ta,we,'curve/rate abs median',np.median(abs(ec)),np.median(abs(er)),'p90',np.quantile(abs(ec),.9),np.quantile(abs(er),.9))
