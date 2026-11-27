exec(open('/tmp/m99_check_candidate.py').read().split('out =')[0])
out=m.solve_batch(ds); tau=out['estimated_taus']; beta=out['hereditary_weights']; t=ds['eval_times']; br=m._bridge(ds['calibration_load_cells']); p,h=m._loads(g,br); x=m._chamber_features(g); d=beta[2]+x@beta[3:]; y=g['strain_curves_obs']-g['strain_0'][:,None]; r=g['instant_rate_obs']; l=g['lab_id']; missing=np.isnan(g['coupon_features_raw']).any(1)
f=-np.expm1(-t[:,None]/tau); G=f*tau-t[:,None]; base=(h[:,None]*beta[:2])@f.T+d[:,None]*t; A=d[:,None,None]*G; target=y-base
age=np.cumsum(ds['past_dt'][::-1])[::-1]-ds['past_dt']; z=p@(np.exp(-age[:,None]/tau)*-np.expm1(-ds['past_dt'][:,None]/tau))
# Each added term tests a physically distinct failure: retained memory, load-drift coupling, or drift saturation.
def correction(q): return q[:,None,None]*G
sets={'fixed':None,'fit_shares':A,'retained_memory':np.concatenate((A,-z[:,None,:]*f),axis=2),'load_drift':np.concatenate((A,correction(d*(h-3))),axis=2),'difference_interaction':np.concatenate((A,correction(d*(z[:,0]-z[:,1]))),axis=2),'rational_drift':np.concatenate((A,correction(d/(1+d*d))),axis=2)}
for name,mat in sets.items():
 E=np.empty_like(y)
 for lab in np.unique(l):
  a=l!=lab; b=~a
  if mat is None: E[b]=A[b]@np.array([.25,.50])-target[b]; continue
  w=np.linalg.lstsq(mat[a].reshape(-1,mat.shape[-1]),target[a].ravel(),rcond=None)[0]; E[b]=mat[b]@w-target[b]
  if name=='fit_shares': print('fit shares without lab',lab,np.round(w,6))
 print(name,'held-lab overall RMSE',round(np.sqrt(np.mean(E**2)),6),'max complete/missing/late',*(round(np.max(abs(e)),6) for e in [E[~missing],E[missing],E[:,-1]]))
err=h*np.sum(beta[:2]/tau)+d-r
print('onset max by lab',[(int(i),round(np.max(abs(err[l==i])),6),round(np.mean(err[l==i]),6)) for i in np.unique(l)])
cal=ds['calibration_load_cells']; raw=cal['bridge_raw']; loads=cal['true_channel_loads']; nominal=np.max(abs(raw),axis=1)<8; inv=m._invert_bridge(raw[nominal],br)
print('nominal calibration axial stress RMSE and max',np.sqrt(np.mean((inv.sum(1)-loads[nominal].sum(1))**2)),np.max(abs(inv.sum(1)-loads[nominal].sum(1))))
