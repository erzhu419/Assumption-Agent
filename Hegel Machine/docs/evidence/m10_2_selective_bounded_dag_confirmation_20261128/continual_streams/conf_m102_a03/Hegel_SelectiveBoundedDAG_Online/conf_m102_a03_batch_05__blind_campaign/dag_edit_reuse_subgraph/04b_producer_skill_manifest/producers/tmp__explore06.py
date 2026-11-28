import numpy as np,pickle,importlib.util,time
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; n=len(y)
p='/tmp/hegel_continual_archive/conf_m102_a03_batch_02_solver.py'; spec=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=m._affine; q=m._quadratic; fit=m._fit
sh,sc=m._calibrate(s['panel_a']); z=(tr['x']-sh)/sc; b=s['panel_b']; B=fit(a(b['u']),b['v']); w=a(z)@B
np.set_printoptions(precision=5,suppress=True)
print('calibration',sh,sc,'transform',B,'panel B RMS',np.sqrt(np.mean((a(b['u'])@B-b['v'])**2)))
print('raw linear residual quantiles',np.quantile(np.linalg.norm(s['panel_a']['v']-sh-s['panel_a']['u']*sc,axis=1),[0,.5,.9,.95,1]));print('traj SVD',np.linalg.svd(T,compute_uv=False));print('y quantiles',np.quantile(y,[0,.1,.5,.9,1]))
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(101).permutation(n),4)]
for nm,F in [('affine',a(z)),('quad',q(z)),('bilinear',np.c_[w,w[:,0]*w[:,2]+w[:,1]*w[:,3]])]:
 print(nm,'in',np.mean(abs(F@fit(F,y)-y)),'out',[round(np.mean(abs(F[j]@fit(F[i],y[i])-y[j])),6) for j in folds for i in [np.setdiff1d(np.arange(n),j)]])
for label,D in [('calibrated',z),('panel b',w)]:
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); now=time.time(); yh=m._readout(D[i],D[j],y[i]); sp=m._state(D[i],D[j],T[i]); print(label,j[:3],round(np.mean(abs(yh-y[j])),6),round(np.mean(abs(sp-T[j])),6),round(time.time()-now,2),flush=True)
