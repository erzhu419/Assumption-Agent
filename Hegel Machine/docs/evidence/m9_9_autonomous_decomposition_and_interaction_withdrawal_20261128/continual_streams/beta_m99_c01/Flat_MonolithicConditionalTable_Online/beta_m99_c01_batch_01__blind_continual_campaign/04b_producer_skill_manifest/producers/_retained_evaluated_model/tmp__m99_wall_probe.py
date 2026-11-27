import json, pickle, runpy
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/'
for r in json.load(open(p+'archive_apparatus_log.json')):
 a=np.asarray(r['facility_telemetry_trace']); print('log',r['completed_campaign_short_name'],a.shape,'mean',a.mean(0).round(4),'end',a[-3:].round(4).tolist())
a=runpy.run_path(p+'beta_m99_c01_s00_solver.py'); it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; ds=it['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; axes=[]
for q,n in [('x',nx),('y',ny)]:
 k,v,h,r=s['kappa_'+q],s['vel_'+q],s['L'+q]/(n-1),s['robin_'+q]; axes.append((a['_axis'](k,v,h,r).toarray(),k,v,h,r))
w=np.array([d['w_pulse'] for d in ds]); modes=np.einsum('pm,mxy->pxy',w,s['source_modes']); rhs=modes.reshape(len(w),-1,order='F').T.copy(); i,j=s['sensor_ij'].T; ids=i+nx*j; obs=np.array([a['_calibrate']({**s,'y_obs_raw':d['y_pulse_raw']}) for d in ds]).T
base=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc')
def pred(params):
 mats=[]
 for (m,k,v,h,r),pp in zip(axes,np.asarray(params).reshape(2,2,3)):
  m=m.copy()
  for l,adj,wrap,(diff,leak,cross) in zip([0,-1],[1,-2],[-1,0],pp):
   D=diff*k[l]/h**2; C=cross*(k[0]+k[-1])/(2*h**2); m[l,l]=D+C+leak*r/h; m[l,adj]=-D; m[l,wrap]=-C
  mats.append(m)
 ax,ay=mats; op=(kron(eye(ny),ax)+kron(ay,eye(nx))+base).tocsc()
 return np.concatenate([splu(op+z*I).solve(rhs)[ids] for z in s['mu_shifts']])
def metrics(par):
 e=pred(par)-obs; return np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1,len(w))**2,axis=1)).round(8).tolist()
for name,row in [('Robin',[1,1,0]),('zero leak',[1,0,0]),('mirror double',[2,0,0]),('half neighbor',[.5,0,0]),('periodic',[1,0,1])]:
 print(name,metrics(row*4))
for omitted in [None,0,1,2]:
 keep=np.arange(len(w))!=omitted if omitted is not None else np.ones(len(w),bool)
 fit=least_squares(lambda p:(pred(p)-obs)[:,keep].ravel(),[1,.05,.05]*4,bounds=([-0.0,-.15,0.]*4,[4,3,3]*4),gtol=1e-12,ftol=1e-12,xtol=1e-12)
 print('omit',omitted,'coefs (neighbor, Robin, wrap) x0,xN,y0,yN:',fit.x.round(6).tolist(),'shift/pulse RMSE',metrics(fit.x))
