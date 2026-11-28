import runpy,pickle,numpy as np
from scipy.optimize import least_squares
A=runpy.run_path('/tmp/hegel_continual_archive/beta_m100_c03_s00_solver.py'); I=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=I['study']; P=I['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in P],axis=1); Y=np.stack([A['_calibrate'](s,p['y_pulse_raw']) for p in P],axis=1)

def response(p):
 axes=[]
 for j,d in enumerate('xy'):
  kap=s['kappa_'+d]; vel=s['vel_'+d]; h=s['L'+d]/(len(kap)-1); rob=s['robin_'+d]; a=A['_axis'](kap,vel,h,rob).tolil()
  for b,(i,k) in enumerate([(0,1),(-1,-2)]):
   diff,leak,wrap=p[6*j+3*b:6*j+3*b+3]; leak0=rob/h; diff0=kap[i]/h**2
   a[i,i]=diff*diff0+leak*leak0+wrap*diff0; a[i,k]=-diff*diff0; a[i,-1 if b==0 else 0]=-wrap*diff0
  axes.append(a.tocsc())
 return A['_responses'](s,axes)[0]@W

base=np.tile([1.,1.,0.],4); lo=np.zeros(12); hi=np.full(12,6.)
def fit(name,expand,start,lower,upper):
 def err(z): return response(expand(z))-Y
 vals=[]
 for k in [-1,0,1,2]:
  keep=np.arange(3)!=k; sol=least_squares(lambda z:err(z)[:,keep].ravel(),start,bounds=(lower,upper),ftol=1e-12,xtol=1e-12,gtol=1e-12); e=err(sol.x); vals.append(np.sqrt(np.mean(e[:,k]**2)) if k>=0 else np.sqrt(np.mean(e**2)))
  if k==-1: print(name, 'params',expand(sol.x).round(7),'by-shift',np.sqrt(np.mean(e.reshape(3,-1)**2,axis=1))); np.save('/tmp/wall_'+name,expand(sol.x))
 print(name,'all,LOPO',vals)
fit('leak_xy',lambda z: np.array([1,z[0],0,1,z[0],0,1,z[1],0,1,z[1],0]),[1,1],0,6)
fit('leak_4',lambda z: np.column_stack([np.ones(4),z,np.zeros(4)]).ravel(),np.ones(4),0,6)
fit('diff_leak',lambda z:np.c_[z.reshape(4,2),np.zeros(4)].ravel(),np.ones(8),0,6)
fit('wrap',lambda z:z,base,lo,hi)
print('calibrated pulse range',Y.min(),Y.max(),'mu',s['mu_shifts']); print('telemetry summaries:')
import json
for x in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(x['facility_telemetry_trace']); print(x['completed_campaign_short_name'],a.mean(0),a.std(0))
