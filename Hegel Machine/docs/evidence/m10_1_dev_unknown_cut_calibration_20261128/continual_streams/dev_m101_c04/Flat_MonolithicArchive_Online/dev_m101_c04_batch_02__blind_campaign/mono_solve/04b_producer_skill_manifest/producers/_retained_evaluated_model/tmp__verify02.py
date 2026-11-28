from importlib.util import spec_from_file_location, module_from_spec
from itertools import combinations_with_replacement
import pickle
import numpy as np
p='/holdout_task/'
spec=spec_from_file_location('batch_solver',p+'batch_solver.py'); m=module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; c=m._calibrate(a)
print('calibration row residuals: four largest',np.sort(np.linalg.norm(np.column_stack([m._calibration_features(a['u'],j)@c[j] for j in range(4)])-a['v'],axis=1))[-4:])
tr=s['train']; x,y,t=map(tr.get,('x','y','traj')); z,zt=m._coordinates(s)
for tag in ('train','test'):
 x0=s[tag]['x']; u=m._unwarp(x0,c); fitted=np.column_stack([m._calibration_features(u,j)@c[j] for j in range(4)]); print(tag,'inverse max error',np.max(abs(fitted-x0))); assert np.max(abs(fitted-x0))<1e-10
sq=np.sum(z*z,axis=1); axes=[np.argsort(abs(z[:,j])) for j in range(4)]
splits=[np.arange(72)]+[np.random.default_rng(i).permutation(72) for i in range(1,9)]+[np.argsort(sq)]+axes
coefs=m._ls(m._scalar(z),y); print('scalar coefficients',coefs)
poly=np.column_stack([np.ones(len(z))]+[np.prod(z[:,k],axis=1) for d in (1,2) for k in combinations_with_replacement(range(4),d)])
for n,i in enumerate(splits):
 f,h=i[:56],i[56:]; pred=m.solve_batch(dict(s,train={k:v[f] if k!='summary' else v for k,v in tr.items()},test={'x':x[h]})); ey=abs(pred['y_hat']-y[h]); et=abs(pred['state_pred']-t[h]); q,r=np.linalg.qr(z[f],mode='reduced'); u,d,v=np.linalg.svd(q.T@t[f],full_matrices=False); svd=z[h]@m._ls(r,(u[:,:2]*d[:2])@v[:2]); print(n,'MAE scalar/traj/worst horizon',*[round(v,6) for v in (ey.mean(),et.mean(),et.mean(0).max())],'alt scalar quadratic/traj unconstrained rank2',round(abs(poly[h]@m._ls(poly[f],y[f])-y[h]).mean(),6),round(abs(svd-t[h]).mean(),6))
