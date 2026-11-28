import glob,importlib.util,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; pa=s['panel_a']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; a=lambda z:np.c_[np.ones(len(z)),z]; fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]; mods={}; zs={'raw':x}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 name=f.split('_')[-2]; sp=importlib.util.spec_from_file_location('m'+name,f); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); mods[name]=m
 try:
  c=m._calibration(pa)
  if hasattr(m,'_decode'):
   z=m._decode(x,c); d=m._decode(pa['v'][:45],c); e=np.linalg.norm(d-pa['u'][:45],axis=1); print(name,'decode calib',np.quantile(e,[0,.5,.9]),'pars',str(c)[:460]); zs[name]=z
  else: print(name,'cal',str(c)[:270])
 except Exception as e: print(name,str(e))
for k,z in zs.items():
 if not np.all(np.isfinite(z)): continue
 quad=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]); designs=[a(z),np.c_[a(z),quad]]; print(k,[(np.mean(abs(D[52:]@fit(D[:52],y[:52])-y[52:])),np.mean(abs(D[52:]@fit(D[:52],tr[:52])-tr[52:]))) for D in designs])
