import importlib.util, pickle, numpy as np
from pathlib import Path

def module(i):
 p=Path('/tmp/hegel_continual_archive')/f'alpha_m100_c04_batch_{i:02}_solver.py'; s=importlib.util.spec_from_file_location(f'm{i}',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; good=np.max(np.abs(y),axis=1)<8
print('calibration:',{k:(type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else None) for k,v in c.items()});print('clean:',good.sum(),'of',len(x),'load range:',x.min(0),x.max(0),'dirty extremes:',np.sort(abs(y).max(1))[-12:]); print('test raw quantiles:',np.quantile(D['test']['past_channel_raw'],[0,.05,.5,.95,1],axis=(0,1)))
u=np.flatnonzero(good); folds=np.array_split(np.random.default_rng(36).permutation(u),5)
for typ in ('linear', 'cubic', 'harmonic'):
 m=module(3 if typ=='cubic' else 5); inv=[]; fwd=[]
 for v in folds:
  a=np.ones(len(x),bool); a[v]=False; a &= good
  if typ=='linear':
   b=np.linalg.lstsq(np.c_[x[a],np.ones(a.sum())],y[a],rcond=None)[0]; recon=np.linalg.solve(b[:4].T,(y[v]-b[4]).T).T.sum(1); yp=np.c_[x[v],np.ones(len(v))]@b
  else:
   b=m._calibrate({'bridge_raw':y[a],'true_channel_loads':x[a]}); recon=m._stress(y[v],b); yp=m._bridge_raw(x[v],b) if typ=='harmonic' else np.einsum('ncf,cf->nc',m._bridge_design(x[v]),b)
  inv.append(recon-x[v].sum(1)); fwd.append(yp-y[v]);
  if typ=='harmonic': print('harmonic shared params',np.round(b[:4],4))
 e=np.concatenate(inv); f=np.concatenate(fwd); print(typ,'stress RMSE',np.sqrt(np.mean(e**2)), 'worst fold',max(np.sqrt(np.mean(t**2)) for t in inv),'abs quantiles',np.quantile(abs(e),[.5,.95,1]),'raw channel RMSE',np.mean(f**2,axis=0)**.5)
