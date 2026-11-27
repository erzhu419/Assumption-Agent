exec(open('/tmp/chamber_probe.py').read().split('for mode in [')[0].split('for label,A in [')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=170)
keys=('coupon_features_raw','ref_coupons','lab_id','stratum')
print('diag keys',list(g)); print('missing',np.argwhere(np.isnan(g['coupon_features_raw'])))
full={k:np.concatenate((tr[k],te[k],g[k])) for k in keys}
separate={k:g[k] for k in keys}
for lab in range(4):
 print('reference',lab,[(np.nanmedian(np.nanmean(q['ref_coupons'][q['lab_id']==lab],0)),np.nanmedian(np.nanstd(q['ref_coupons'][q['lab_id']==lab],0))) for q in (tr,te,g,full)])
Z=z(tau); D0=D.copy(); K=lambda q:-np.expm1(-t[:,None]/q)
def predict(p, D, reset=False):
 q=np.exp(p[:2]); h=p[2:4]; m=np.r_[0.,p[4]] if reset else p[4:6]; a,b,logdur,off=p[-4:]; dur=np.exp(logdur)
 E=(H[:,None]*h-Z*m)*th[:2]; F=-dur*np.expm1(-t/dur)
 return E@K(q).T+D[:,None]*(a*t+b*F)+off*t, E@(1/q)+D*(a+b)+off
start=np.r_[np.log(tau),1.,1.,.0,.25,.23,.79,0.,0.]; ix=np.arange(len(Y)); target=np.c_[Y,R]
def residual(p,D):
 u,v=predict(p,D); return np.c_[u-Y,v-R]
p=least_squares(lambda p:residual(p,D).ravel(),start,loss='soft_l1',f_scale=.01).x
err=residual(p,D)
print('params',p); print('id lab stratum hold Zslow drift missing curve-early/mid/late rate residual'); print(np.c_[ix,g['lab_id'],g['stratum'],H,Z[:,1],D,np.isnan(g['coupon_features_raw']).sum(1),err[:,[0,5,11,12]]]);
if 'aux_chamber_telemetry' in g: print('telemetry',g['aux_chamber_telemetry'])
# Independent per-coupon instantaneous drift, with transient and drift time shape fixed.
u,v=predict(p,np.zeros(len(Y))); F=np.exp(p[-2])*-np.expm1(-t/np.exp(p[-2])); design=np.r_[p[-4]*t+p[-3]*F,p[-4]+p[-3]]; independent=(target-np.c_[u,v])@design/(design@design); print('inferred drift, standard, difference',np.c_[independent,D,independent-D]); print('drift shifts by lab',[(lab,np.mean((independent-D)[g['lab_id']==lab])) for lab in np.unique(g['lab_id'])]);
# Chamber feature correction candidates: per-lab reference alignment versus the strain-derived drift shifts.
raw=g['coupon_features_raw'];
for label, ref in [('train/test', {k:full[k][:-len(Y)] for k in keys}),('all',full),('diag only',g)]:
 W=raw.copy()
 for lab in np.unique(g['lab_id']):
  mask=g['lab_id']==lab; controls=ref['ref_coupons'][ref['lab_id']==lab]; sc=np.nanmedian(np.nanstd(controls,0)); mu=np.nanmedian(np.nanmean(controls,0)); W[mask]=(W[mask]-mu)/(sc if sc>0 else 1)
 W=np.where(np.isfinite(W),W,xg); DD=np.c_[np.ones(len(Y)),W]@th[2:]; pp=least_squares(lambda p:residual(p,DD).ravel(),start,loss='soft_l1',f_scale=.01).x; ee=residual(pp,DD); print('reference variant',label,score(ee[:,:-1]),score(ee[:,-1]))
