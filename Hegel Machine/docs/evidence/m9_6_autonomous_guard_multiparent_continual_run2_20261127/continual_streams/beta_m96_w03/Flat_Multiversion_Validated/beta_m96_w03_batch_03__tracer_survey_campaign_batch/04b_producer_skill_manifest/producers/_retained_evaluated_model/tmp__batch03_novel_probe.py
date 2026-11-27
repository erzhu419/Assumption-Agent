import importlib.util, json, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/beta_m96_w03_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
it=pickle.load(open('public_beta_m96_w03_batch_03.pkl','rb'))[0]; s,g=it['study'],it['gt']; archive=json.loads(m._HISTORY.read_text())
A,_=m._forward(s); D=np.column_stack((A,s['Z_nuisance'])); rows=np.asarray(s['row_bank_id']); obs=m._calibrated_observations(s); S=g['S_tilde']; r=g['r_y']; ridge=s['source_ridge']; cap=s['u_box']
for ep, data in [(archive[2]['episode_short_name'],archive[2]),('public',s['sensor_calibration_sweeps'])]:
 bank,x,y=(np.asarray(data[k]) for k in ['bank_id','ref_signal','raw_readout']); truth=m._calibration_coefficients(data); raw=truth[rows,0]*obs+truth[rows,1]
 for mode in ['bank-0-checks','bank-1-check','wing-0-checks']:
  worst=np.zeros(4)
  for b in range(2 if mode.startswith('wing') else 4):
   missing=(rows//2==b if mode.startswith('wing') else rows==b)
   coef=np.linalg.lstsq(D[~missing],obs[~missing],rcond=None)[0]; pred=D@coef
   selections=np.flatnonzero(bank==b) if mode=='bank-1-check' else [None]
   for j in selections:
    inferred=truth.copy()
    for k in np.unique(rows[missing]):
     q=rows==k
     if j is None: inferred[k]=m._fit_affine(pred[q],raw[q])
     else:
      px=pred[q]-x[j]; gain=px@(raw[q]-y[j])/(px@px); inferred[k]=gain,y[j]-gain*x[j]
    calibrated=(raw-inferred[rows,1])/inferred[rows,0]; X,v=m._profile(s,A,calibrated)
    for budget,w0 in zip(s['budget_grid'],g['weights_by_budget']):
     w,nu=m._box_simplex(X,v,ridge,cap,float(budget)); c=m._certificate(X,v,ridge,cap,float(budget),w,nu); e=S@w-r; grad=S.T@e+ridge*w; obj=(e@e+ridge*(w@w))/2; lower=obj-grad@w+nu*budget+cap*np.minimum(grad-nu,0).sum()
     worst=np.maximum(worst,[np.max(abs(inferred-truth)),np.max(abs(w-w0)),obj-min(lower,c['dual_lower_bound']),np.max(abs(grad-nu-c['lam_lower']+c['lam_upper']))])
  print(ep,mode,'worst coefficient / weight / oracle gap / stationarity:',*worst)
