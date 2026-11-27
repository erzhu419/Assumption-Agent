exec(open('/tmp/chamber_probe.py').read().split('a=cal')[0])
from nonlinear_bridge import fit_bridge
iv, keep = fit_bridge(cal)
a=cal['true_channel_loads']; b=cal['bridge_raw']; e=[]
for j in np.flatnonzero(keep):
    sub=np.arange(len(a))!=j
    ii,_=fit_bridge(dict(true_channel_loads=a[sub],bridge_raw=b[sub])); e.append(ii(b[j:j+1])[0]-a[j])
print('calibration rejected',np.flatnonzero(~keep),'LOO axial RMSE, worst:',np.sqrt(np.mean(np.sum(e,1)**2)),np.max(abs(np.sum(e,1))))
def spec(z,x):
    return iv(z['past_channel_raw']).sum(-1),iv(z['hold_channel_raw']).sum(-1),np.c_[np.ones(len(x)),x]
xt,xe=m._features(tr,te); st=spec(tr,xt); tau,th=m._fit(st,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);print('tau, theta:',tau,th)
y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']
ref={k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; _,x=m._features(ref,g); p,h,X=spec(g,x)
c,rr=m._branches((p,h,X),dt,t,tau); dr=X@th[2:]; B=c@th[:2]; R=rr@th[:2];
print('diagnostic reference-pool transfer:')
for name,F in [('linear',t),('exp1',-np.expm1(-t)),('no drift',0*t)]:
 ec=y-B-dr[:,None]*F; er=r-R-(dr if name!='no drift' else 0); print(name,'curve/rate RMSE',np.sqrt(np.mean(ec**2)),np.sqrt(np.mean(er**2)), 'by time mean',np.round(ec.mean(0),4))
np.set_printoptions(precision=4,suppress=True,linewidth=150)
print('diagnostic hold, first/last history, drift, observed rate, baseline rate, final residual');print(np.c_[h,p[:,0],p[:,-1],dr,r,R+dr,(y-B-dr[:,None]*t)[:,-1]])
print('archive log keys and fields')
import json
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({k:v for k,v in row.items() if k!='facility_telemetry_trace'})
