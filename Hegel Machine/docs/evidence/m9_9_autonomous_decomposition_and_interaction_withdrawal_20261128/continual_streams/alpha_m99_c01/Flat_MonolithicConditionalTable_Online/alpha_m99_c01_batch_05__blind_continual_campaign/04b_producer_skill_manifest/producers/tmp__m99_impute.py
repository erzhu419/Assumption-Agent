exec(open('/tmp/m99_chamber.py').read().split('for name,A,V in')[0])
import json
from pathlib import Path
idx=json.loads(Path('/holdout_task/public_asset_index.json').read_text()); print('index top',list(idx)); print('index details',[(k,str(v)[:260]) for k,v in idx.items()]);
log=json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text()); print('log top',list(log) if isinstance(log,dict) else len(log)); es=log if isinstance(log,list) else next(v for v in log.values() if isinstance(v,list) and v and isinstance(v[-1],dict));
for e in es:
 print('log entry',[(k,(str(v)[:220] if not isinstance(v,dict) else list(v))) for k,v in e.items()]);
keys=('coupon_features_raw','ref_coupons','lab_id','stratum'); empty={k:g[k][:0] for k in keys}; xgd,_=m._features(g,empty); xtd,_=m._features(te,empty); raw=g['coupon_features_raw']; missing=~np.isfinite(raw); complete=~missing.any(1)
Q=Y-(hg[:,None]*beta[:2])@F[:,:2].T; curve=np.r_[t,1]+np.r_[np.c_[(f*tau-t[:,None])],np.zeros((1,2))]@np.array([.25,.5]); dmeas=Q@curve/(curve@curve); dlocal=beta[2]+xgd@beta[3:];
print('diag implied fills (assuming one missing):')
for i,j in zip(*np.where(missing)):
 print(i,j,'lab/stratum',g['lab_id'][i],g['stratum'][i],'local',xgd[i,j],'joint train-test',np.unique(np.vstack([x,xt])[~np.isfinite(np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]))[:,j],j]),'implied',xgd[i,j]+(dmeas[i]-dlocal[i])/beta[3+j],'bench + diag fill',xg[i,j]);
def prefills(parts,align):
 X=[]; L=[]; S=[]
 for q in parts:
  xr=q['coupon_features_raw'].copy(); labs=q['lab_id']; refs=q['ref_coupons'];
  for l in np.unique(labs):
   sel=labs==l; R=refs[sel] if align=='local' else np.vstack([p['ref_coupons'][p['lab_id']==l] for p in parts]); xr[sel]=(xr[sel]-np.median(R.mean(0)))/np.median(R.std(0));
  X.append(xr); L.extend(labs); S.extend(q['stratum']);
 return np.vstack(X),np.asarray(L),np.asarray(S)
for name,parts in [('g',[g]),('tr',[tr]),('te',[te]),('tr+te',[tr,te]),('all',[tr,te,g]),('g+te',[g,te])]:
 for al in ('local','joint') if len(parts)>1 else ('local',):
  xr,l,s=prefills(parts,al); _,inv,counts=np.unique(np.c_[l,s],axis=0,return_inverse=True,return_counts=True); w=1/counts[inv]; print(name,al,'weighted mean',np.nansum(xr*w[:,None],0)/(np.isfinite(xr).T@w),'plain mean',np.nanmean(xr,0));
print('deployment local vs joint drift range', np.quantile(beta[2]+xtd@beta[3:],[0,.5,1]),np.quantile(beta[2]+xt@beta[3:],[0,.5,1]));
