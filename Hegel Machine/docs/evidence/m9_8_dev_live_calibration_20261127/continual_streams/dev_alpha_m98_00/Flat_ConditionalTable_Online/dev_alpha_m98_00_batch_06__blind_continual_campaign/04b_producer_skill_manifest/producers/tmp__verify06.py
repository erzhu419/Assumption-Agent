import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr,te,g=d['train'],d['test'],i['campaign_diagnostics'];times=d['eval_times'];out=m.solve_batch(d);tau,theta=out['estimated_taus'],out['hereditary_weights']; print('material',tau,theta)
c=d['calibration_load_cells'];co=m._fit_bridge(c);x,b=c['true_channel_loads'],c['bridge_raw']; print('calibration gross',np.where(np.linalg.norm(m._bridge_forward(x,co)-b,axis=1)>.01)[0]);
empty=lambda q:{k:v[:0] for k,v in q.items()}; hg=m._invert_bridge(g['hold_channel_raw'],co).sum(axis=-1)
for name,xg in [('diagnostic self',m._aligned_features(g,empty(g))),('train-diagnostic',m._aligned_features(tr,g)[len(tr['lab_id']):]),('test-diagnostic',m._aligned_features(te,g)[len(te['lab_id']):])]:
 y,r=m._chamber(tau,theta,times,hg,xg); e=y+g['strain_0'][:,None]-g['strain_curves_obs'];f=r-g['instant_rate_obs'];print(name,'RMSE',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(f**2)),'worst coupons',max(np.sqrt(np.mean(e**2,1))),max(abs(f))); print('rate errors by coupon',np.round(f,4))
for name,other in [('train only',empty(tr)),('train plus test',te)]:
 X=m._aligned_features(tr,other)[:len(tr['lab_id'])]; q=m._coupon(tr,X,co);Y,R=m._design(tau,d['past_dt'],times,q); print(name,'at current parameters: rate rms',np.sqrt(np.mean((R@theta-tr['instant_rate_obs'])**2)))
for q in (tr,te,g):
 for k in ('past_channel_raw','hold_channel_raw'):
  bq=q[k]; xx=m._invert_bridge(bq,co).reshape(-1,4); print('bridge convergence',k,abs(m._bridge_forward(xx,co)-bq.reshape(-1,4)).max())
