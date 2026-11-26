exec(open('/tmp/alpha_ep02_cv.py').read().split('for lab in [-1,')[0])
cal=d['calibration_load_cells']
uncalibrated=dict(d, calibration_load_cells=dict(cal, true_stress=cal['cmd_stress']))
a_uncal,_,_,_=m._prepare(uncalibrated)
for label, model in [('calibrated',a),('command stress',a_uncal)]:
 stats=[]
 for lab in np.unique(labs):
  g=labs==lab
  tau,w=m._fit(model,y,complete & ~g)
  raw=(model.matrix(np.log(tau))@w)[g]
  pred=m._update_drift(raw,y[g,-1],complete[g],times)
  err=np.abs(y[g,:-1]-pred[:,:-1]); ls=tr['stratum'][g]
  row=(np.quantile(err,.9),max(np.median(err[ls==s,-1]) for s in np.unique(ls)),np.quantile(err[~complete[g]],.9) if np.any(~complete[g]) else np.nan)
  stats.append(row)
  print(label,lab,'curve p90 / worst stratum late median / imputed curve p90',np.round(row,5))
 print('worst',label,np.round(np.nanmax(stats,axis=0),5))
