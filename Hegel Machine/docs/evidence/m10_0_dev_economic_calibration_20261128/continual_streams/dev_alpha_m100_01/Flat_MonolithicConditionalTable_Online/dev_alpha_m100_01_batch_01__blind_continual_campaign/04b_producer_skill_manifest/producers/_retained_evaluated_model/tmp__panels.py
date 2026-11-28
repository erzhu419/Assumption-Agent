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
for name,P in [('tr',tr),('te',te),('diag',c),('joint',{k:np.r_[tr[k],te[k]] for k in ['ref_coupons','lab_id','coupon_features_raw']})]:
 print(name,'missing',np.isnan(P['coupon_features_raw']).sum(axis=0))
 for l in np.unique(P['lab_id']):
  m=P['lab_id']==l; refs=P['ref_coupons'][m]; X=P['coupon_features_raw'][m]; print(l,m.sum(),'med mean/std',np.median(refs.mean(0)),np.median(refs.std(0)),'raw mean/std',np.nanmean(X,0),np.nanstd(X,0))
