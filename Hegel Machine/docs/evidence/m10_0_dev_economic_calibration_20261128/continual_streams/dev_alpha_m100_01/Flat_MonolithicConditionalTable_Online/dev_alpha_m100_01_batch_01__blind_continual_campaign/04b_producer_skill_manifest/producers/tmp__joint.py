import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from dev_alpha_m100_01_s00_solver import _features,_bridge,_loads,_fit,_design
np.set_printoptions(precision=5,suppress=True,linewidth=140)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,c=d['train'],d['test'],i['campaign_diagnostics']; n=len(tr['lab_id']); dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; b=_bridge(cal)
R,Y=cal['bridge_raw'],cal['true_channel_loads']; print('bridge',b); print('train hold history loads ranges',[(v.min(),v.max()) for v in _loads(tr,b)]); print('diag hold history loads ranges',[(v.min(),v.max()) for v in _loads(c,b)]);
from itertools import combinations_with_replacement
for p in range(1,4):
 A=np.column_stack([np.ones(len(R))]+[np.prod(R[:,k],axis=1) for q in range(1,p+1) for k in combinations_with_replacement(range(4),q)]); pinv=np.linalg.pinv(A); E=Y-A@(pinv@Y); E=E/(1-np.einsum('ij,ji->i',A,pinv))[:,None]; print('cal LOO degree',p,'channel rms',np.mean(E**2,axis=0)**.5,'total rms',np.mean(E.sum(axis=1)**2)**.5)
s=cal['single_channel_sweeps'];print('sweeps:');
for lab in np.unique(s['lab_id']):
 m=s['lab_id']==lab; x0,y0=s['true_stress'][m],s['cmd_stress'][m]; print(lab, np.array([x0,y0]).T[::2], [np.std(y0-np.polyval(np.polyfit(x0,y0,p),x0)) for p in range(1,5)])
x=_features(tr,te); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; ta,th=_fit(*_loads(tr,b),x[:n],dt,t,yc,yr); print('taus theta',ta,th)
xc=_features(dict((k,tr[k]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')),c)[n:]; past,hold=_loads(c,b); cy=c['strain_curves_obs']-c['strain_0'][:,None]; cr=c['instant_rate_obs'];
def metrics(c,r,y,z):
 e=c-y; f=r-z;return 'curve/rate MAE',np.mean(abs(e)),np.mean(abs(f)),'curve/rate median',np.median(abs(e)),np.median(abs(f)),'90%',np.quantile(abs(e),.9),np.quantile(abs(f),.9)
import json
from itertools import product
Y=np.c_[cy,cr]; Xs={'diag':_features(c,c)[:len(cy)],'joint':_features(tr,te)}; n=len(cy)
# Diagnostic reference panel is independent; also try applying the actual deployment normalizer.
ref=np.r_[tr['ref_coupons'],te['ref_coupons']]; labs=np.r_[tr['lab_id'],te['lab_id']]; xx=c['coupon_features_raw'].copy(); raw=np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]; fill=np.array([x[~np.isfinite(raw[:,j]),j][0] if (~np.isfinite(raw[:,j])).any() else 0 for j in range(4)])
for l in np.unique(labs):
 p=ref[labs==l]; m=c['lab_id']==l; xx[m]=(xx[m]-np.median(p.mean(0)))/np.median(p.std(0))
Xs['joint']=np.where(np.isfinite(xx),xx,fill)
for label,X in Xs.items():
 r=th[2]+X@th[3:]; print('ref',label,'drift range',r.min(),r.max())
 for tim,slopes,drift in product(['fixed','free'],['index','full'],['off','on']):
  F=np.c_[np.ones(n),r] if slopes=='index' else np.c_[np.ones(n),X]; N=len(F[0]);
  def model(logta):
   tau=np.exp(logta); Q=np.r_[-np.expm1(-t[:,None]/tau),[1/tau]]; known=(hold[:,None]*th[:2])@Q.T
   A=[F[:,None,:]*Q[None,:,b,None] for b in range(2)]
   if drift=='on': A.append(F[:,None,:]*np.r_[t,1.][None,:,None]);
   return np.concatenate(A,axis=2),known
  def fit(m):
   def profile(logta,ret=False):
    a,k=model(logta); D=a[m].reshape(m.sum()*Y.shape[1],-1); w=np.linalg.lstsq(D,(Y[m]-k[m]).ravel(),rcond=None)[0]; P=a@w+k
    return (P,w) if ret else (P[m]-Y[m]).ravel()
   lt=np.log(ta) if tim=='fixed' else least_squares(profile,np.log(ta),bounds=(np.log([.3,1.2]),np.log([1.,12.]))).x
   return *profile(lt,True),np.exp(lt)
  p,w,tau=fit(np.ones(n,dtype=bool)); o=np.zeros_like(Y)
  for l in np.unique(c['lab_id']):
   m=c['lab_id']==l; o[m]=fit(~m)[0][m]
  E=abs(o-Y); print(tim,slopes,drift,'tau',tau,'coef',w,'rms',np.sqrt(np.mean((p-Y)**2)), 'worst LOLO MAE', np.max([E[c['lab_id']==l,:-1].mean() for l in np.unique(c['lab_id'])]),np.max([E[c['lab_id']==l,-1].mean() for l in np.unique(c['lab_id'])]))
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 A=np.asarray(row['facility_telemetry_trace']);print({k:v for k,v in row.items() if k!='facility_telemetry_trace'},'telemetry means',A.mean(0),'std',A.std(0))
for k,p in [('train',tr),('test',te),('diag',c)]:
 A=p.get('aux_chamber_telemetry'); print(k,'aux',None if A is None else (A.mean(0),A.std(0)))
