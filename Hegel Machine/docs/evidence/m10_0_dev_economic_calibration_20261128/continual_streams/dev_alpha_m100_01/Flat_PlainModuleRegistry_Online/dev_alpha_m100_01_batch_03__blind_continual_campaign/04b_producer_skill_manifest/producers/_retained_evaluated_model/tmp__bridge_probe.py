import numpy as np, pickle
from itertools import combinations_with_replacement
r=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('items',len(r));d=r[0]['dataset'];c=d['calibration_load_cells'];print('calibration', {k:(getattr(v,'shape',None) or list(v) if isinstance(v,dict) else getattr(v,'shape',type(v).__name__)) for k,v in c.items()});print('top-level',list(d));v=d.get('chamber_verification_trials');print('verification',type(v),{k:getattr(s,'shape',None) for k,s in v.items()} if isinstance(v,dict) else v is None);s=c['single_channel_sweeps'];print('sweeps',{k:np.shape(v) for k,v in s.items()}); x=np.asarray(c['true_channel_loads']);y=np.asarray(c['bridge_raw']);print('x range',x.min(0),x.max(0),'y range',np.quantile(y,[0,.05,.95,1],axis=0));print('x rows',x[:10]); print('sweep starts',{k:np.asarray(v).ravel()[:8].tolist() for k,v in s.items()});
def matrix(x,deg):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,it],axis=1) for p in range(1,deg+1) for it in combinations_with_replacement(range(4),p)])
def fit(A,y):
 b=np.linalg.lstsq(A,y,rcond=None)[0];w=np.ones(len(y))
 for _ in range(25):
  e=np.linalg.norm(A@b-y,axis=1);w=1/(1+(e/max(2*np.median(e),1e-9))**2); t=np.sqrt(w);b=np.linalg.lstsq(t[:,None]*A,t[:,None]*y,rcond=None)[0]
 return b,w
for j in range(1,5):
 a=matrix(x,j); b,w=fit(a,y);pred=np.empty_like(y);idx=np.random.default_rng(31).permutation(len(x));
 for ts in np.array_split(idx,5):
  tr=np.setdiff1d(idx,ts);pred[ts]=a[ts]@fit(a[tr],y[tr])[0]
 e=np.linalg.norm(pred-y,axis=1);tail=np.max(np.abs((x-np.median(x,axis=0))/np.std(x,axis=0)),axis=1); edge=tail>=np.quantile(tail,.8);print('degree',j,'parameters',len(b),'OOF e quantiles',np.quantile(e,[.5,.8,.9,.95,.98,1]),'edge quantiles',np.quantile(e[edge],[.5,.8,.9,1]),'weights<.1',np.where(w<.1)[0]);
 if j in (2,3):
  for k in range(4):print(k,[(i,round(z,5)) for i,z in enumerate(b[:,k]) if abs(z)>.002]);
