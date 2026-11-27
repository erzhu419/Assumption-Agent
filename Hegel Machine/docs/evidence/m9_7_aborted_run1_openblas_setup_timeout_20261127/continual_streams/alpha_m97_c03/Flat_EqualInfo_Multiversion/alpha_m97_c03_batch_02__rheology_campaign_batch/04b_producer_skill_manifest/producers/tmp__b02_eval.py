import importlib.util, pickle, numpy as np
base='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',base+'alpha_m97_c03_batch_02_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open(base+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d=p['dataset']; g=p['gt']; tr,te=d['train'],d['test']; out=s.solve_alpha_m97_c03_batch_02(d)
def rm(a): return np.sqrt(np.mean(np.square(a)))
print('parameters actual',out['estimated_taus'],out['hereditary_weights'],'reference',g['taus'],g['theta'])
print('parameter abs err',abs(out['estimated_taus']-g['taus']),abs(out['hereditary_weights']-g['theta']))
x=s._align(tr,te); oracle=s._design(np.log(g['taus']),s._inputs(te,x[len(tr['lab_id']):],s._calibrate(d['calibration_load_cells'])),d['past_dt'],d['eval_times'])[0]@g['theta']; oracle[:,:-1]+=te['strain_0'][:,None]
e=out['predicted_strain']-g['te_clean']; er=out['initial_rate_pred']-g['te_true_rate0']; oe=oracle[:,:-1]-g['te_clean']; oer=oracle[:,-1]-g['te_true_rate0']; comp=np.isfinite(te['coupon_features_raw']).all(1)
for lab in range(4):
 q=te['lab_id']==lab
 print('clean-test lab',lab,'all/completes/incompletes with n, strain,rate,oracle strain,oracle rate',[(a.sum(),rm(e[a]),rm(er[a]),rm(oe[a]),rm(oer[a])) for a in (q,q&comp,q&~comp) if a.any()])
 q=tr['lab_id']==lab
 fold=dict(d,train={k:v[~q] for k,v in tr.items()},test={k:np.concatenate((tr[k][q],v)) for k,v in te.items()}); f=s.solve_alpha_m97_c03_batch_02(fold); n=q.sum(); err=f['predicted_strain'][:n]-tr['strain_curves_obs'][q]; rr=f['initial_rate_pred'][:n]-tr['instant_rate_obs'][q]; complete=np.isfinite(tr['coupon_features_raw'][q]).all(1)
 print('left-out lab',lab,'fit tau',f['estimated_taus'],'median abs across complete rows worst horizon',np.max(np.median(abs(err[complete]),axis=0)),'complete rate median abs',np.median(abs(rr[complete])))
