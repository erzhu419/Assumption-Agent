import numpy as np, pickle, importlib.util
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m100_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];print('diagnostics',type(p.get('campaign_diagnostics')).__name__)
d=p['dataset']; c=d['calibration_load_cells'];x,y=c['true_channel_loads'],c['bridge_raw'];b,k=m._fit_bridge(y,x);r=np.linalg.norm(m._bridge_forward(x,b)-y,axis=1);print('bridge',b,'kept',k.sum(),'quantiles',np.quantile(r,[0,.5,.8,.9,.95,1]),'largest',[(int(i),np.round(r[i],4)) for i in np.argsort(r)[-16:]])
for key in ('train','test'):
 for f in ('past_channel_raw','hold_channel_raw'):
  raw=d[key][f]; ld=m._load_from_raw(raw,b);err=np.max(np.abs(m._bridge_forward(ld,b)-raw)); print(key,f,'load bounds',ld.min(),ld.max(),'inverse residual',err)
a=np.random.default_rng(240).permutation(len(x));
for val in np.array_split(a,5):
 tr=np.setdiff1d(a,val); b,k=m._fit_bridge(y[tr],x[tr]); e=np.linalg.norm(m._bridge_forward(x[val],b)-y[val],axis=1);print('fold',k.sum(),np.round(np.sort(e),4))
