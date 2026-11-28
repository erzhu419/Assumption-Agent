import importlib.util,pickle,numpy as np
p='/holdout_task'; spec=importlib.util.spec_from_file_location('candidate',p+'/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open(p+'/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,traj=tr['x'],tr['y'],tr['traj']; t=s['eval_times'];
z=m._calibrate(s['panel_a'],x);g,h=m._operators(s['panel_c'],z); a,b=m._readout(g,h); coef=m._ls(b[:52],np.c_[y,m._ls(m._time(t,[.42,1.82]),traj.T).T][:52]); np.set_printoptions(precision=3,suppress=True,linewidth=120);print('coefs',coef[:9].T);print('cross matrices',coef[9:].T.reshape(3,4,4)); print('dot scalar',m._ls(a[:52],y[:52]));
worst=np.zeros(3)
def run(name,te):
 global worst
 keep=np.setdiff1d(np.arange(72),te); test=dict(s,train={**tr, **{k:tr[k][keep] for k in ('x','y','traj')}},test={'x':x[te]}); pred=m.solve_batch(test); e=np.abs(pred['state_pred']-traj[te]); score=[np.mean(abs(pred['y_hat']-y[te])),e.mean(),e.mean(0).max()]; worst=np.maximum(worst,score); print(name,np.round(score,6));
run('specified',np.arange(52,72))
for seed in range(5): run('random-'+str(seed),np.random.default_rng(seed).choice(72,20,replace=False))
for j in range(4):
 for sign in (-1,1):run('boundary-'+str(j)+'-'+str(sign),np.argsort(sign*z[:,j])[-20:])
print('worst',worst)
