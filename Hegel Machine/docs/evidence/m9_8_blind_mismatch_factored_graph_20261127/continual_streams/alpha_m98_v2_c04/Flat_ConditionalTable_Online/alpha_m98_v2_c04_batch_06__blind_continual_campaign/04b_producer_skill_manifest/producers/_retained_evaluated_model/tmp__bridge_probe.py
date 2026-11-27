import pickle,numpy as np
from itertools import combinations_with_replacement
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; c=d['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']
print('cal min/max raw',R.min(0),R.max(0),'true',L.min(0),L.max(0)); print('rows raw/true',np.round(np.c_[R,L][:12],3));
for k,b in [('train',d['train']),('test',d['test']),('diag',it['campaign_diagnostics'])]:
 r=np.vstack((b['hold_channel_raw'],b['past_channel_raw'].reshape(-1,4)));print(k,np.quantile(r,[0,.01,.99,1],axis=0).round(3))
for x,y,way in [(L,R,'forward'),(R,L,'inverse')]:
 for deg in [1,2,3,4,5]:
  for mix in [False,True] if deg>1 else [True]:
   terms=[z for p in range(1,deg+1) for z in combinations_with_replacement(range(4),p) if mix or len(set(z))==1]
   a=np.column_stack([np.ones(len(x))]+[x[:,z].prod(1) for z in terms]); z=np.linalg.lstsq(a,y,rcond=None)[0]; res=a@z-y; lev=np.sum(np.linalg.svd(a,full_matrices=False)[0]**2,axis=1); e=res/(1-lev[:,None]); print(way,deg,mix,'loo med/rms/max',np.round(np.median(abs(e),axis=0),5),np.round(np.mean(e**2,axis=0)**.5,5),np.round(abs(e).max(0),5),'total rms',np.mean(e.sum(1)**2)**.5)
   if deg in [1,3] and not (mix and deg==3): print('coefs',np.round(z,4));
