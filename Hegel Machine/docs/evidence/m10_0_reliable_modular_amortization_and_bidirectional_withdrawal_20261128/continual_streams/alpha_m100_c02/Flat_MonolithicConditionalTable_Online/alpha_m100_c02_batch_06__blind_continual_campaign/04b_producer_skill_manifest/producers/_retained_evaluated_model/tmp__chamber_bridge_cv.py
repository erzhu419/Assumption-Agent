import sys,pickle,importlib,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
mods=[importlib.import_module('alpha_m100_c02_batch_'+i+'_solver') for i in ('02','03','05')]
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; l=c['true_channel_loads']; r=c['bridge_raw']; n=len(l)
print('stress quantiles',np.quantile(l,[0,.1,.5,.9,1],axis=0)); print('nonzero channels per row',np.unique(np.count_nonzero(l,axis=1),return_counts=True))
for m in mods:
 try:
  summaries=[]; residual=np.empty_like(l); total=np.empty(n); sweep=c['single_channel_sweeps']; sl=sweep['true_stress']; sr=sweep['cmd_stress']; ch=sweep['lab_id'];
  for k,ids in enumerate(np.array_split(np.random.default_rng(81).permutation(n),5)):
   keep=np.ones(n,bool); keep[ids]=False
   if m is mods[-1]:
    s=np.arange(len(sl))%5!=k; a=m._fit_bridge(l[keep],r[keep],ch[s],sl[s],sr[s]);
   else: a=m._fit_bridge(dict(c,true_channel_loads=l[keep],bridge_raw=r[keep]))
   e=m._invert_bridge(r[ids],a)-l[ids]; residual[ids]=e; total[ids]=e.sum(axis=1); summaries.append((np.median(abs(e.sum(axis=1))),np.quantile(abs(e.sum(axis=1)),.8)))
  print(m.__name__, 'fold median/p80',np.round(summaries,5),'channel abs q',np.quantile(abs(residual),[.5,.8,.9]),'total abs q',np.quantile(abs(total),[.5,.8,.9]));
  if m is mods[-1]: print('fit coefficients',np.round(m._fit_bridge(l,r,ch,sl,sr),6))
  else: print('fit coefficients',np.round(m._fit_bridge(c),6))
 except Exception as e: print(type(e).__name__,str(e))
