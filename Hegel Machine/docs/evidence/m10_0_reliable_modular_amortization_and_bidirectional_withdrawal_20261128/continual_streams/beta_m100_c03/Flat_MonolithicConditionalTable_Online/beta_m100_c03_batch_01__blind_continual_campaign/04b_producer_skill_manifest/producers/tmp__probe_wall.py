import importlib.util,pickle,json,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=item['study'];P=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]).T;Y=np.array([m._calibrate(s,p['y_pulse_raw']) for p in item['campaign_diagnostics']]).T

def axes(p,kind):
 out=[]
 for i,d in enumerate('xy'):
  kap=s['kappa_'+d];v=s['vel_'+d];h=s['L'+d]/(len(kap)-1);r=s['robin_'+d];a=m._axis(kap,v,h,r).tolil()
  for j,(b,k,o) in enumerate([(0,1,-1),(-1,-2,0)]):
   z=2*i+j; D=kap[b]/h**2; F=p[z]*r/h; C=0
   if kind in ('diff','wrap'): D*=p[4+z]
   if kind=='wrap': C=p[8+z]*kap[b]/h**2
   a[b,b]=D+F+C; a[b,k]=-D; a[b,o]=-C
  out.append(a.tocsc())
 return out

def err(p,kind): return m._responses(s,axes(p,kind))[0]@P-Y

def report(label,e,p=None): print(label, np.sqrt(np.mean(e*e)),np.sqrt(np.mean(e.reshape(3,-1,e.shape[1])**2,axis=1)).round(7), None if p is None else p.round(8),flush=True)
for val in [0,0.125,0.25,0.5,1]: report('fixed Robin '+str(val),err(np.full(4,val),'rob'))
for kind,x in [('rob',[0.1]*4),('diff',[0.1]*4+[1]*4),('wrap',[0.1]*4+[1]*4+[0]*4)]:
 x=np.array(x);f=least_squares(lambda p: err(p,kind).ravel(),x,gtol=1e-12,ftol=1e-12,xtol=1e-12);report(kind,err(f.x,kind),f.x)
 for test in range(3):
  train=np.arange(3)!=test; f=least_squares(lambda p: err(p,kind)[:,train].ravel(),x,gtol=1e-11);report('held '+str(test),err(f.x,kind)[:,[test]],f.x)
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 t=np.array(row['facility_telemetry_trace']); print('telemetry',row['completed_campaign_short_name'],'mean',t.mean(0),'std',t.std(0))
