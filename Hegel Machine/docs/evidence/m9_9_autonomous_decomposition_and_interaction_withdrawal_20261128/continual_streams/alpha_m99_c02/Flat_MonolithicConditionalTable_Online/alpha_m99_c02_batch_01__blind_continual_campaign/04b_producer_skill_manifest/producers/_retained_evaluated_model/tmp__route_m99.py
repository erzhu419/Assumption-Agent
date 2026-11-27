import contextlib,io,json
with contextlib.redirect_stdout(io.StringIO()):
 from probe_m99 import *
p,h=m._loads(g,b); lab=g['lab_id']; n=len(h)
T=np.vstack((np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1.]))
Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
M=h[:,None]-m._basis(ta,p,h,gx,dt,t)[1][:,:2]*ta
Y=Y-(h[:,None]*th[:2])@T[:,:2].T
# Test no history, partial history, and diagnostic feature normalization independently.
fields=('coupon_features_raw','ref_coupons','lab_id','stratum')
def combine(*sets): return {k:np.concatenate([s[k] for s in sets]) for k in fields}
X={'train+test+diag':gx,'train+diag':m._features(tr,g)[1],'diag':m._features(g,g)[1],'test+diag':m._features(te,g)[1]}
# Normalize diagnostic features with train+test panel and fill using train+test only.
j=combine(tr,te); raw=j[fields[0]].copy(); q=g[fields[0]].copy()
for a in np.unique(lab):
 panel=j['ref_coupons'][j['lab_id']==a]; mu=np.median(panel.mean(0)); sd=np.median(panel.std(0)); q[lab==a]=(q[lab==a]-mu)/sd; raw[j['lab_id']==a]=(raw[j['lab_id']==a]-mu)/sd
_,cell,count=np.unique(np.c_[j['lab_id'],j['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/count[cell]; fill=np.nansum(raw*w[:,None],axis=0)/(w@np.isfinite(raw)); X['train+test panel']=np.where(np.isnan(q),fill,q)
Z=Y@np.linalg.pinv(T.T); print('raw inferred amplitudes after hold (columns); missing feature count'); print(np.c_[Z,np.isnan(q).sum(1)])
for name,x in X.items():
 D=th[2]+x@th[3:]; print('alignment',name)
 for variant,F in [('route',np.c_[np.ones(n),D]),('route+memory',np.c_[np.ones(n),D,M]),('free feature weights',np.c_[np.ones(n),x])]:
  E=np.zeros_like(Y)
  for a in np.unique(lab):
   keep=lab!=a; C=np.linalg.lstsq(F[keep],Y[keep],rcond=None)[0]@np.linalg.pinv(T.T); E[~keep]=F[~keep]@C@T.T-Y[~keep]
  C=np.linalg.lstsq(F,Y,rcond=None)[0]@np.linalg.pinv(T.T); print(variant,'C',C,'worst lab',np.max([np.sqrt([np.mean(E[lab==a,:-1]**2),np.mean(E[lab==a,-1]**2)]) for a in np.unique(lab)],axis=0),'complete/missing',*[np.sqrt(np.mean(E[k]**2)) for k in [~np.isnan(q).any(1),np.isnan(q).any(1)]])
  if variant=='route': print('signed residual by coupon',np.c_[D,(F@C@T.T-Y)[:,-1],lab])
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 A=np.array(row['facility_telemetry_trace']); print(row['completed_campaign_short_name'],A.mean(0),A.std(0))
