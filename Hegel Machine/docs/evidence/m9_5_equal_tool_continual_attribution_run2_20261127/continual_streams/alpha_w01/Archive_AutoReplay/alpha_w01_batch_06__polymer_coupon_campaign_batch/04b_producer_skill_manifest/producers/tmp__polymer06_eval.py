import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_w01_batch_06_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=pickle.load(open(p+'public_alpha_w01_batch_06.pkl','rb'))[0];d=r['dataset'];gt=r['gt'];tr=d['train'];te=d['test'];t=d['eval_times'];dt=d['past_dt'];ys=d['yield_stress']
rmse=lambda e: np.sqrt(np.mean(np.asarray(e)**2))
o=m.solve_alpha_w01_batch_06(d);maps=m._load_cells(d['calibration_load_cells']);print('map difference',maps-np.column_stack((gt['stress_gains'],gt['stress_biases'])))
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]: print(a,np.round(o[a],7),np.round(gt[b],7))
for name,block,target,rate in [('public test',te,gt['te_clean'],gt['te_true_rate0'])]:
 ec=o['predicted_strain']-target;er=o['initial_rate_pred']-rate;eo=block['instant_rate_obs']-rate
 for l in range(4):
  for cross in (0,1):
   for missing in (0,1):
    sel=(block['lab_id']==l)&(np.asarray(gt['te_cross'])==cross)&(np.isnan(block['coupon_features_raw']).any(1)==missing)
    if sel.any(): print(name,l,cross,missing,sel.sum(),*(round(v,6) for v in (rmse(ec[sel]),rmse(er[sel]),rmse(eo[sel]))))
for lab in range(4):
 val=tr['lab_id']==lab
 take=lambda mask:{k:v[mask] for k,v in tr.items()}
 q=m.solve_alpha_w01_batch_06({**d,'train':take(~val),'test':take(val)})
 ev=q['predicted_strain']-tr['strain_curves_obs'][val];rv=q['initial_rate_pred']-tr['instant_rate_obs'][val]; inp=m._inputs(take(val),np.zeros((val.sum(),4)),maps,dt,ys);cross=(inp[-2]!=0)|(inp[-1]!=0)
 for c in (False,True):
  sel=cross==c
  print('LOLO',lab,int(c),sum(sel),*(round(v,6) for v in (rmse(ev[sel]),rmse(rv[sel]),rmse(ev[sel,0]),rmse(ev[sel,-1]))))
