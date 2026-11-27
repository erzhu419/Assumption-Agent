exec(open('/tmp/b03_diag.py').read().split('p=m._fit(h,tr);')[0])
import json
hist=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
print('archive sweep keys',list(hist[0]))
lab=np.r_[tr['lab_id'],te['lab_id']]; strat=np.r_[tr['stratum'],te['stratum']]; raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); pooled=np.vstack([x,xe]); cell=pooled.copy(); missing=~np.isfinite(raw)
for l,s in np.unique(np.c_[lab,strat],axis=0):
 use=(lab==l)&(strat==s); obs=np.where(missing[use],np.nan,pooled[use]); mu=np.nanmean(obs,axis=0)
 cell[use]=np.where(missing[use]&np.isfinite(mu),mu,pooled[use])
miss=missing[:len(x)].any(axis=1)
for name,X in [('balanced pooled',pooled),('within cell',cell)]:
 H=m._Hold(d['past_dt'],d['eval_times'],tr,X[:len(x)],cal); oof=np.empty_like(y)
 for l in range(4):
  use=tr['lab_id']==l; fit=m._fit(H,tr,~use); a,_=H.design(fit[:2]); oof[use]=y[use]-a[use]@fit[2:]
 print(name)
 for label,use in [('complete',~miss),('missing',miss),('wing A',tr['lab_id']<2),('wing B',tr['lab_id']>=2)]:
  r=oof[use]; print(label,use.sum(),'curve RMSE',np.sqrt(np.nanmean(r[:,:-1]**2)),'last medAE',np.nanmedian(abs(r[:,-2])),'rate MAE',np.nanmean(abs(r[:,-1])))
print('largest terminal errors (row, missing count, residual):',[(int(i),int(missing[i].sum()),round(float(oof[i,-2]),3)) for i in np.argsort(np.nan_to_num(abs(oof[:,-2])))[-10:]])
C=d['calibration_load_cells']; sample={k:np.asarray(v).copy() for k,v in C.items()}; ix=np.flatnonzero(sample['lab_id']==2); sample['true_stress'][ix[:3]]+=0.2
print('three EMI check drift at +/- 2.5',np.max(abs((np.asarray(m._calibrate(sample))-np.asarray(cal)).T@np.array([[-2.5,2.5],[1.,1.]]))))
keep=(sample['lab_id']!=2); keep[ix[-1]]=True
print('one check fallback',np.asarray(m._calibrate({k:v[keep] for k,v in sample.items()}))[:,2])
