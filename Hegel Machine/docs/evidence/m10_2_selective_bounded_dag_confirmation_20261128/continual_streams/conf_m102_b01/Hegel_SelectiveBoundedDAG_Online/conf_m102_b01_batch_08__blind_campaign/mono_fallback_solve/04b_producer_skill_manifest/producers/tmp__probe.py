import pickle, numpy as np, sys, importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']
def imp(path):
 sp=importlib.util.spec_from_file_location('m',path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=imp('/tmp/hegel_continual_archive/conf_m102_b01_batch_06_solver.py'); aff=m._aff; ls=m._ls
print('times',s['eval_times'])
for a in [52, 0, 20, 40]:
 te=np.arange(a,a+20); tr=np.setdiff1d(np.arange(72),te); sub=dict(s,train={k:t[k][tr] for k in ('x','y','traj')},test={'x':t['x'][te]}); p=m.solve_batch(sub); print('archive',a,np.mean(abs(p['y_hat']-t['y'][te])),np.mean(abs(p['state_pred']-t['traj'][te])))
pa=s['panel_a']; u,v=pa['u'],pa['v']; z=m._calibrate(pa,v); print('A inverse errors',np.round(np.sort(np.linalg.norm(z-u,axis=1)),4)); print('Z ranges',m._calibrate(pa,t['x']).min(0),m._calibrate(pa,t['x']).max(0));
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[panel]; u,v=p['u'],p[key]; print(panel,key)
 for name,f in [('archive c1',m._pairmap(u)),('archive c2',m._even(u)),('affine',aff(u)),('quad',aff(np.c_[u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]])),('odd',aff(np.c_[u,u**3,u**5])),('evenodd',aff(np.c_[u,u**2,u**3,u**4,u**5]))]:
  r=v-f@ls(f,v); h=(f*np.linalg.pinv(f).T).sum(1); print(name,np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/np.maximum(1-h[:,None],1e-10))**2)))
