from pathlib import Path
import sys,pickle,numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,'/tmp'); import base_solver as m
exec(Path('/tmp/extension.py').read_text(),m.__dict__)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; results=[]
with threadpool_limits(limits=1):
 for seed in (None,0,1,2):
    p=np.arange(72) if seed is None else np.random.default_rng(seed).permutation(72); i,j=p[:52],p[52:]; t=tr['traj'][i]; R=np.linalg.svd(t,full_matrices=False)[2][:3]; target=np.c_[tr['y'][i],t@R.T]; coords=m._coordinates(s,tr['x'][p]); k=m._kernels(coords,52); g=m._graph_kernels(s,coords,52); ext=np.concatenate((k,g)); base=m._fit(k,target); more=m._fit(ext,target); joint=m._fit(ext,target,separate=False);
    out=[]
    for a in (base,more,(base+more)/2,(base+more+joint)/3):
        e=np.abs(a[:,0]-tr['y'][j]); f=np.abs(a[:,1:]@R-tr['traj'][j]);out.append([e.mean(),f.mean(),f.mean(0).max()]);
    results.append(out);print(seed,np.round(out,5),flush=True)
r=np.asarray(results); scores=r[:,:,:2].mean(axis=(0,2))+.5*r[:,:,:2].max(axis=(0,2)); j=int(np.argmin(scores));
if j and (r[0,j,:2]<=r[0,0,:2]+.008).all() and r[:,j,:2].max()<=r[:,0,:2].max()+.005:
    src=Path('/tmp/base_solver.py').read_text(); statement={1:'result=_fit(ext,target)',2:'result=(_fit(ks,target)+_fit(ext,target))/2',3:'result=(_fit(ks,target)+_fit(ext,target)+_fit(ext,target,separate=False))/3'}[j];src=src.replace('result=_fit(ks,target)','ext=np.concatenate((ks,_graph_kernels(study,coords,n)))\n        '+statement);Path('/holdout_task/batch_solver.py').write_text(src+Path('/tmp/extension.py').read_text());print('promoted',j)
else: print('retained baseline')
