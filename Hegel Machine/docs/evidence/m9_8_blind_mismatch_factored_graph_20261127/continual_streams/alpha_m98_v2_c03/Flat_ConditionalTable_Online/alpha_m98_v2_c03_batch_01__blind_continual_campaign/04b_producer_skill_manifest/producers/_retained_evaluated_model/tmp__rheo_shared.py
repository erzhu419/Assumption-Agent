exec(open('/tmp/rheo_probe.py').read().split("print('train RMSE'")[0])
import json
P,H,D=c; n=len(H); B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]])[:,:2]; A=np.linalg.lstsq(B,yd.T,rcond=None)[0].T; Q=A-H[:,None]*w[:2]; print('SVD chamber excess',np.linalg.svd(Q,compute_uv=False)); print('coupling direction',np.linalg.lstsq(Q[:,0:1],Q[:,1],rcond=None)[0]); print('excess residual',np.std(Q[:,1]-10.5*Q[:,0])); keys=('lab_id','stratum','coupon_features_raw','ref_coupons');
configs={'train+test+diag':D[:,1:],'train+diag':m._features(tr,dg)[-n:],'test+diag':m._features(te,dg)[-n:],'diag only':m._features(dg,dg)[-n:]}
for name,X in configs.items():
 R=np.c_[np.ones(n),X]@w[2:]; fits=[]
 for inter in (False,True):
  Z=np.c_[np.ones(n),R] if inter else R[:,None]; C=np.linalg.lstsq(Z,Q,rcond=None)[0]; E=Z@C-Q; fits.append((C.round(4).tolist(),np.std(E,axis=0).round(4).tolist(),np.round([np.mean(E[dg['lab_id']==i,1]) for i in np.unique(dg['lab_id'])],3).tolist()))
 print(name, fits)
R=D@w[2:]; print('diagnostic row data: lab, missing, R, inferred R from slow, H'); print(np.c_[dg['lab_id'],np.isnan(dg['coupon_features_raw']).sum(1),R,Q[:,1]/3,H].round(3));
for entry in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print({k:v for k,v in entry.items() if k!='facility_telemetry_trace'}); print('trace mean',np.mean(entry['facility_telemetry_trace'],axis=0).round(3))
