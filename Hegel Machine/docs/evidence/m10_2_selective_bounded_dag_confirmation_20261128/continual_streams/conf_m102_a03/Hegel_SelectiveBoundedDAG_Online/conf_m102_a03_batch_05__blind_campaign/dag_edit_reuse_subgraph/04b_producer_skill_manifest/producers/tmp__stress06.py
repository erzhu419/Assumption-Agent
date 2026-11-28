import importlib.util,pickle,numpy as np,itertools
from scipy.linalg import solve
from scipy.spatial.distance import cdist
p='/holdout_task/node_00.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; n=len(x); co=m._sensor_calibration(s['panel_a']); z=(x-co[0])/co[1]; w=m._readout_coordinates(s['panel_b'],z)
folds=[np.arange(n-20,n),*np.array_split(np.random.default_rng(101).permutation(n),4),*np.array_split(np.random.default_rng(807).permutation(n),4),*np.array_split(np.argsort(np.sum(z*z,axis=1)),4)]
for j in folds:
 i=np.setdiff1d(np.arange(n),j); ss={**s,'train':{k:t[k][i] for k in ('x','y','traj')},'test':{'x':x[j]}}; f=m.run_node({'__input__':ss}); yp=f['readout_test']@solve(f['readout_train'],y[i],assume_a='pos'); cp=np.concatenate([np.outer(solve(k,tr[i]@v,assume_a='pos'),v) for k,v in zip(f['state_train'],f['modes'])]); print('BASE',len(j),round(np.mean(abs(yp-y[j])),6),round(np.mean(abs(f['zt']@cp-tr[j])),6))
def kernels(v,alpha):
 g=v@v.T/v.shape[1]; ds=(v[:,None,:]-v[None,:,:])**2; return g, np.exp(-alpha*ds.mean(2)),np.exp(-alpha*ds).mean(2)
opts=list(itertools.product((.15,.35,.5,.8,1.3),(.0,.3,1.,3.),(0.,.15,.5),(.003,.01,.03,.06,.1)))
errs=np.empty((len(opts),len(folds)))
for fi,j in enumerate(folds):
 i=np.setdiff1d(np.arange(n),j); ww=(w-w[i].mean(0))/w[i].std(0); zz=(z-z[i].mean(0))/z[i].std(0); K={al:kernels(ww,al) for al in (.15,.35,.5,.8,1.3)}; gz,rz,az=kernels(zz,1.2)
 for oi,(al,ad,mix,r) in enumerate(opts):
  g,rb,ab=K[al]; k=1+g+rb+ad*ab+mix*(gz+az); errs[oi,fi]=np.mean(abs(k[np.ix_(j,i)]@solve(k[np.ix_(i,i)]+r*np.eye(len(i)),y[i],assume_a='pos')-y[j]))
for oi in np.argsort(errs[:,1:9].mean(1)+.45*errs[:,1:9].max(1)+.20*errs[:,9:].mean(1))[:12]: print('OPT',opts[oi],np.round(errs[oi],5))
