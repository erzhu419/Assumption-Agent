import glob,importlib.util,pickle,numpy as np,copy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=np.concatenate([s['train']['y'][:,None],s['train']['traj']],1)
for k in ('eval_times','telemetry_vec'):print(k,s[k])
print('train x quantiles',np.quantile(s['train']['x'],[0,.1,.5,.9,1],axis=0));print('outputs',np.round(p[:8],5)); print('outputs sv',np.linalg.svd(p-p.mean(0),compute_uv=False))
for k in ('panel_a','panel_b','panel_c'):
 d=s[k];u=d['u'];print(k,'u range',u.min(),u.max());
 for n,v in d.items():
  if n=='u':continue
  A=np.c_[np.ones(len(u)),u]; C=np.linalg.lstsq(A,v,rcond=None)[0]; r=np.linalg.norm(v-A@C,axis=1);print(n,'range',v.min(),v.max(),'linear res',np.quantile(r,[0,.25,.5,.75,.9,1]),'coeff',np.round(C,3));print('sample',np.round(np.c_[u,v][:3],4))
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  spec=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  q=copy.deepcopy(s);q['train']={k:v[:52] for k,v in q['train'].items()};q['test']={'x':s['train']['x'][52:]};o=m.solve_batch(q); yh=np.c_[o['y_hat'],o['state_pred']];print(f.split('/')[-1],np.round(np.mean(abs(yh-p[52:]),axis=0),6))
 except Exception as e:print(f,repr(e))
