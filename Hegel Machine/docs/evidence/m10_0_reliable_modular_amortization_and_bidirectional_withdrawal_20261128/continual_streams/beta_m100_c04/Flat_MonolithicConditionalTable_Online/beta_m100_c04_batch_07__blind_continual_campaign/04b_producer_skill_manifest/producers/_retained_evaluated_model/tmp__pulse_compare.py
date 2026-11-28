import glob,importlib.util,pickle,numpy as np,itertools
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]); Y=np.array([p['y_pulse_raw'] for p in pulses]); print('pulse raw range',Y.min(),Y.max()); print('observations raw range',np.min(s['y_obs_raw']),np.max(s['y_obs_raw'])); print('pulse weights',W)
mods={}
for path in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 key=path.split('_')[-2]; spec=importlib.util.spec_from_file_location('m'+key,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[key]=m
f=mods['06']; Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; print('nuisance projection',Z.shape)
laws=[(a,b) for a in (0,.15,.35,.5,1,1.5,2) for b in (0,.15,.35,.5,1,1.5,2) if a==b or a==1 or b==1]+['periodic']; ss={str((a,b)):f._forward(s,(a,b))[0] for a,b in itertools.product(laws,repeat=2)}
for key,m in mods.items():
 try:
  y=np.array([m._calibrate(dict(s,y_obs_raw=r)) for r in Y]); print(key,'calibrated range',y.min(),y.max())
  results=[]
  for law,S in ss.items():
   d=W@S.T-y; c=d-(d@Q)@Q.T; results.append((np.linalg.norm(c),np.linalg.norm(d),law,np.round(np.linalg.norm(c,axis=1),6)))
  print(*sorted(results)[:3],sep='\n')
  if key!='06':
   S=m._forward(s)[0]; d=W@S.T-y; print('native wall',np.linalg.norm(d-(d@Q)@Q.T),np.linalg.norm(d))
 except Exception as e: print(key,type(e).__name__,e)
