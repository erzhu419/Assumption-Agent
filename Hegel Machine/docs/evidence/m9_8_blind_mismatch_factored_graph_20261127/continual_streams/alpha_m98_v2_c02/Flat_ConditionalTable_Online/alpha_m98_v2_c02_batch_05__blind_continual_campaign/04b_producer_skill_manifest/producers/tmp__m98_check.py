import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; x,xt=m._standardize(tr,te); stress=m._stress_calibration(d['calibration_load_cells']); dt,t=d['past_dt'],d['eval_times']; ys=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; out=m.solve_batch(d)
print('taus',out['estimated_taus'],'theta',out['hereditary_weights']); print('deployment ranges',[(k,float(np.min(out[k])),float(np.max(out[k]))) for k in ('predicted_strain','initial_rate_pred')]);
for lab in np.unique(tr['lab_id']):
    keep=tr['lab_id']!=lab
    def design(mask): return m._design({k:v[mask] for k,v in tr.items()},x[mask],stress,dt,t)
    a,b=m._fit(design(keep),ys[keep],yr[keep]); z,r=design(~keep)(a); e=z@b-ys[~keep]; er=r@b-yr[~keep]; print('LOLO',lab,'taus',a.round(5),'curve median abs',np.median(abs(e)),'curve q90 abs',np.quantile(abs(e),.9),'early/late median abs',np.median(abs(e[:,0])),np.median(abs(e[:,-1])),'rate RMSE',np.sqrt(np.mean(er**2)))
a,b=out['estimated_taus'],out['hereditary_weights']; z=0
for h,v in zip(dt,stress(te['past_channel_raw']).T):
    w=np.exp(-h/a); z=w*z+(1-w)*v[:,None]
g=stress(te['hold_channel_raw'])[:,None]-z; r=b[2]+xt@b[3:]; q=te['strain_0'][:,None]+(g*b[:2])@(-np.expm1(-t[None,:]/a[:,None]))+r[:,None]*t
np.testing.assert_allclose(out['predicted_strain'],q,rtol=1e-12,atol=1e-12); np.testing.assert_allclose(out['initial_rate_pred'],(g/a)@b[:2]+r,rtol=1e-12,atol=1e-12)
