import importlib.util, json, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
p='/tmp/hegel_continual_archive/beta_m99_c02_batch_01_solver.py'
sp=importlib.util.spec_from_file_location('prev',p); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=i['study'];ds=i['campaign_diagnostics'];z=s['Z_nuisance'];q=np.linalg.qr(z)[0]
w=np.column_stack([d['w_pulse'] for d in ds]); y=np.column_stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds]);
base=[]
for a in 'xy':
 h=s['L'+a]/(s['n'+a]-1); k=s['kappa_'+a];v=s['vel_'+a];base.append((m._axis(k,v,h,0).toarray(),k,v,h,s['robin_'+a]))
def prediction(p,wrap=False):
 axes=[]
 for j,(A,k,v,h,r) in enumerate(base):
  A=A.copy()
  for b,(edge,near,far) in enumerate(((0,1,-1),(-1,-2,0))):
   t=2*j+b;loc=p[t]*k[edge]/h**2;leak=p[4+t]*r/h;wr=p[8+t]*k[edge]/h**2 if wrap else 0
   A[edge,near]=-loc;A[edge,far]=-wr;A[edge,edge]=loc+leak+wr
  axes.append(csc_matrix(A))
 return m._responses(s,axes)[0]@w

def err(p,wrap):
 e=prediction(p,wrap)-y;return e-q@(q.T@e)
for mode in ('paired','wrapped'):
 wr=mode=='wrapped';init=np.r_[np.ones(4),np.zeros(4+(4 if wr else 0))];
 for hold in (-1,0,1,2):
  keep=np.arange(3)!=hold
  res=least_squares(lambda p:err(p,wr)[:,keep].ravel(),init,bounds=(np.r_[np.zeros(4),np.full(4,-.25),np.zeros(4 if wr else 0)],np.full(len(init),10.)),max_nfev=180,ftol=1e-11,xtol=1e-11,gtol=1e-11)
  e=err(res.x,wr);print(mode,hold,res.x.round(6),'pulse rms',np.sqrt(np.mean(e**2,axis=0)).round(8))
  if hold<0:print('shift x pulse rms',np.sqrt(np.mean(e.reshape(3,32,3)**2,axis=1)).round(8),'nuisance coefficient',np.linalg.lstsq(z,y-prediction(res.x,wr),rcond=None)[0].round(6))
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(e['facility_telemetry_trace']);print('apparatus',e['completed_campaign_short_name'],'mean',a.mean(0).round(3),'tail',a[-1].round(3))
