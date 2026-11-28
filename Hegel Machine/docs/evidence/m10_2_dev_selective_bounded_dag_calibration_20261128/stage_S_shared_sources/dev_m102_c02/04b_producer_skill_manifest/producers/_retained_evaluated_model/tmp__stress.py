import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z=(d[k] for k in ('x','y','traj')); n=len(y); idx=np.arange(n); rng=np.random.default_rng(1)
slices=[('designated',idx[52:])]+[(f'random{k}',j) for k,j in enumerate(np.array_split(rng.permutation(n),4))]
for k in range(x.shape[1]):
 order=np.argsort(x[:,k]); slices.extend([(f'coordinate{k}_low',order[:14]),(f'coordinate{k}_high',order[-14:])])
slices.append(('radial_edge',np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-14:]))
np.set_printoptions(precision=6,suppress=True)
for name,j in slices:
 i=np.setdiff1d(idx,j); tr={k:d[k][i] for k in ('x','y','traj')}; tr['summary']=np.array([y[i].mean(),y[i].std(),z[i,-1].mean(),z[i,-1].std()]); out=m.solve_batch(dict(s,train=tr,test={'x':x[j]})); e=abs(out['state_pred']-z[j]); raw=m._affine(x[j])@np.linalg.lstsq(m._affine(x[i]),z[i],rcond=None)[0]; print(name,round(abs(out['y_hat']-y[j]).mean(),6),'state',round(e.mean(),6),'worst horizon',round(e.mean(0).max(),6),'raw state',round(abs(raw-z[j]).mean(),6))
b=np.linalg.lstsq(m._quadratic(x),y,rcond=None)[0]; I,J=np.triu_indices(x.shape[1]); H=np.zeros((x.shape[1],x.shape[1])); H[I,J]=b[5:]; H=(H+H.T)/2; print('quadratic coefficients',b,'eigensystem',np.linalg.eigh(H)); print('projected state singular values',np.linalg.svd(m._affine(x)@np.linalg.lstsq(m._affine(x),z,rcond=None)[0],compute_uv=False))
