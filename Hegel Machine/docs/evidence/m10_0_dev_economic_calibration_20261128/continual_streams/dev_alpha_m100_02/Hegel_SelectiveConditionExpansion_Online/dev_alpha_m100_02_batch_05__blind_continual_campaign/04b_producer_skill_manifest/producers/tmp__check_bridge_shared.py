import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from upstream_transducer import _fit_bridge,_forward,_invert
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
 d=item['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; s=c['single_channel_sweeps']; axis=(x!=0).sum(1)==1
 clean=np.max(abs(y-np.median(y,axis=0)),axis=1)<8
 for label,folds in [('random',np.array_split(np.random.default_rng(123).permutation(len(x)),5)),('radial',np.array_split(np.argsort(np.max(abs(x),axis=1)),5)),('axis/mixed',[np.flatnonzero(axis),np.flatnonzero(~axis)])]:
  scores=[]
  for ix in folds:
   f=np.zeros(len(x),bool); f[ix]=True
   sw=np.ones(len(s['lab_id']),bool)
   for k in range(4):
    for t in x[f&axis&(x[:,k]!=0),k]: sw &= ~((s['lab_id']==k)&np.isclose(s['true_stress'],t,atol=1e-9))
   model=_fit_bridge(dict(c,true_channel_loads=x[~f],bridge_raw=y[~f],single_channel_sweeps={k:v[sw] for k,v in s.items()}))
   use=f&clean; r=_forward(x[use],model)-y[use]; e=_invert(y[use],model)-x[use]
   scores.append([np.max(abs(r)),np.max(abs(e)),np.max(abs(e.sum(1)))])
  print(label,'worst raw/load/load sum',np.max(scores,axis=0)); assert max(z[1] for z in scores)<.005
 m=_fit_bridge(c); print('parameters',m); print('worst clean raw residual',np.max(abs(_forward(x[clean],m)-y[clean])))
