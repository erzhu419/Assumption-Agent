import importlib.util, pickle, numpy as np
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',path+'alpha_m96_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
p=pickle.load(open(path+'public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); dt,t=d['past_dt'],d['eval_times']; x=s._align(tr,te)
def predict(tau,theta,a):
 b,_,rate,_=s._basis(tau,dt,t,a[0],a[1]); drift=a[2]@theta[2:]; return b@theta[:2]+drift[:,None]*t,rate@theta[:2]+drift
rms=lambda z:float(np.sqrt(np.mean(z*z)))
out=s.solve_alpha_m96_w02_s00(d)
truth=s._inputs(te,x[n:],g['stress_gains'],g['stress_biases']); oracle,rate0=predict(g['taus'],g['theta'],truth)
print('public tau abs error',abs(out['estimated_taus']-g['taus']),'weight max abs error',max(abs(out['hereditary_weights']-g['theta'])))
for a,b,label in ((out['predicted_strain'],oracle+te['strain_0'][:,None],'strain'),(out['initial_rate_pred'],rate0,'rate')):
 print('worst lab imputed oracle RMSE',label,max(rms((a-b)[te['lab_id']==j]) for j in range(4)))
for seed in (31,45,100):
 rng=np.random.default_rng(seed); w=np.array([.67,.42,.075,.14,-.10,.09,-.07]); gains=np.array([1.04,.96,1.08,.94]); biases=np.array([.09,-.06,.03,-.10]); cc=d['calibration_load_cells']; lab=cc['lab_id']; cmd=cc['cmd_stress']; cy=gains[lab]*cmd+biases[lab]+rng.normal(0,.0001,len(cmd)); cy[[0,11,23,26,32]] += [.4,-.3,.5,-.45,.3]; ge,be=s._calibration(dict(cc,true_stress=cy)); print('EMI gain/bias max error',max(abs(ge-gains)),max(abs(be-biases)))
 for tt in (np.array([.21,7.85]),np.array([.31,1.12]),np.array([1.7,6.7])):
  aa=s._inputs(tr,x[:n],gains,biases); ss,rr=predict(tt,w,aa); mask=rng.random(ss.shape)<.08; ss+=rng.normal(0,.008,ss.shape); ss[mask]+=.12*rng.standard_cauchy(mask.sum()); rr+=rng.normal(0,.008,len(rr)); tau,theta=s._fit(dt,t,s._inputs(tr,x[:n],ge,be),ss,rr); print(seed,'stress taus',tt,'estimated',tau,'weight max abs error',max(abs(theta-w)))
