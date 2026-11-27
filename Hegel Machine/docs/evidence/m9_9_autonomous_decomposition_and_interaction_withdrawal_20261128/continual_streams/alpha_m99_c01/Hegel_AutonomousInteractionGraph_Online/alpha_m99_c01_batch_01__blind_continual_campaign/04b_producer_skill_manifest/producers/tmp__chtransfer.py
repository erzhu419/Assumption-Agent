exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
import json
x=json.load(open('/holdout_task/public_asset_index.json'))
for k,v in x.items(): print('INDEX',k,str(v)[:250])
_,xs=m._features(di,di)
r=beta[2]+xs@beta[3:]; base=hd[:,None]*(basis(tau)[:,:2]@beta[:2]); Q=Y-base
u,sv,v=np.linalg.svd(Q,full_matrices=False); s=Q@v[0]; s*=np.sign(np.corrcoef(r,s)[0,1]); print('reset remainder sv',sv[:6]);
for name,a in [('diag',di),('test',te),('train',tr)]:
 print(name,'keys',list(a),'refs summary',np.round(np.c_[np.mean(a['ref_coupons'],axis=0),np.std(a['ref_coupons'],axis=0)],4).tolist());
 print(name,'feature NaNs',np.isnan(a['coupon_features_raw']).sum(axis=0),'by-lab refs',[(l,round(np.median(a['ref_coupons'][a['lab_id']==l].mean(0)),5),round(np.median(a['ref_coupons'][a['lab_id']==l].std(0)),5)) for l in np.unique(a['lab_id'])]);
for name,F in [('linear',np.c_[r]),('drift square',np.c_[r,r*r]),('hold interaction',np.c_[r,hd*r]),('feature direction',xs),('load and raw-channel direction',np.c_[r,di['hold_channel_raw']]),('history',np.c_[r,pd])]:
 F=np.c_[np.ones(len(r)),F]; er=np.empty_like(Y); b=np.linalg.lstsq(F,Q,rcond=None)[0];
 for l in np.unique(di['lab_id']):
  k=di['lab_id']==l; er[k]=Q[k]-F[k]@np.linalg.lstsq(F[~k],Q[~k],rcond=None)[0]
 print(name,'LOLO rms/max',np.round([np.sqrt(np.mean(er[:,:-1]**2)),np.sqrt(np.mean(er[:,-1]**2)),np.max(abs(er[:,:-1])),np.max(abs(er[:,-1]))],5),'modes',np.round(np.linalg.lstsq(basis(tau),b.T,rcond=None)[0].T,5).tolist())
e=s-np.c_[np.ones(len(r)),r]@np.linalg.lstsq(np.c_[np.ones(len(r)),r],s,rcond=None)[0]; print('scalar remainder rows lab,stratum,missing,hold,drift,error',np.round(np.c_[di['lab_id'],di['stratum'],np.isnan(di['coupon_features_raw']).sum(1),hd,r,e],5).tolist())
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('facility',[(row['completed_campaign_short_name'],np.mean(row['facility_telemetry_trace'],0).round(4).tolist(),np.std(row['facility_telemetry_trace'],0).round(4).tolist()) for row in log]);
