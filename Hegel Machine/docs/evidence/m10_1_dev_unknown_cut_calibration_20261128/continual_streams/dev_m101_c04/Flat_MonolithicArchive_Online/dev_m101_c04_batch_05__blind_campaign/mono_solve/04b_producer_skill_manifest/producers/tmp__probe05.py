import sys,pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
p='/tmp/hegel_continual_archive/dev_m101_c04_batch_04_solver.py'; sp=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; ls=m._ls; aff=m._affine
cal=m._calibrate(s['panel_a']); print('cal',cal); a=s['panel_a']; er=np.sqrt(sum((a['v'][:,j]-m._calibration_features(a['u'],j)@cal[j])**2 for j in range(4))); print('cal sorted residual',np.sort(er)); u=m._unwarp(x,cal)
b=s['panel_b']; fun=lambda u,p: np.sign(u)*np.maximum(abs(u)-p[0],0)**p[1]; f=lambda p:aff(fun(b['u'],p)); opt=least_squares(lambda p:(f(p)@ls(f(p),b['v'])-b['v']).ravel(),[.35,1.5],bounds=([.1,.8],[.7,2.3]),gtol=1e-13); print('b',opt.x,np.linalg.norm(opt.fun),ls(f(opt.x),b['v'])); z=aff(fun(u,opt.x))@ls(f(opt.x),b['v']); old=aff(m._scalar(z)); coef=ls(old,y); res=y-old@coef; print('old readout',coef,'y/res std',y.std(),res.std(),'z ranges',z.min(0),z.max(0)); print('high residual rows',np.c_[u,z,y,res][np.argsort(abs(res))[-12:]].round(3));
rng=np.random.default_rng(912); folds=[np.arange(72)]+[rng.permutation(72) for _ in range(5)];
for name,q in [('raw',x),('calibrated',u),('panel_b',z),('b_nonlinearity',fun(u,opt.x))]:
 for d in range(1,5):
  F=PolynomialFeatures(d).fit_transform(q); errs=[np.mean(abs(F[ix[56:]]@ls(F[ix[:56]],y[ix[:56]])-y[ix[56:]])) for ix in folds]; print(name,d,np.round(errs,5),np.std(y-F@ls(F,y)))
print('traj residual per time',np.std(t['traj']-aff(z)@ls(aff(z),t['traj']),axis=0)); print('traj coefficient',ls(aff(z),t['traj']))
