import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c02_batch_03_solver as m
items=pickle.load(open('/holdout_task/public_batch.pkl','rb')); item=items[0]; s=item['study']; d=item['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in d],axis=1); Y=np.stack([m.calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in d],axis=1)
print('items',len(items)); print('wall inputs',s['nx'],s['ny'],s['Lx'],s['Ly'],s['robin_x'],s['robin_y']); print('x k/v',s['kappa_x'],s['vel_x']); print('y k/v',s['kappa_y'],s['vel_y']); print('other grids',[(t['study']['nx'],t['study']['ny'],t['campaign_diagnostics'] is not None) for t in items]); print('keys',list(s)); orig=m.axis_operator
h=s['Lx']/(s['nx']-1); k=s['kappa_x']; v=s['vel_x']; intra=k[[0,-1]]/h**2; face=(k[[0,-1]]+k[[1,-2]])/(2*h*h); wrap=(k[0]+k[-1])/(2*h*h); print('stencil scales',intra,face,wrap,'velocity/h',v[[0,-1]]/h)
def err(p):
 def axis(k,v,h,losses):
  a=orig(k,v,h,losses).tolil()
  if k is s['kappa_x']:
   n=len(k)
   for j,i,b,o in ((0,0,1,n-1),(1,n-1,n-2,0)):
    f,q,loss=p[j],p[2+j],p[4+j]; a[i,b]=-f; a[i,o]=-q; a[i,i]=f+q+loss
  return a.tocsc()
 m.axis_operator=axis
 return m.chamber_response(s,np.zeros(4))[0]@W-Y
x=np.r_[intra,[wrap,wrap],0.,0.]; rule=np.r_[intra+[max(-v[0],0)/h,max(v[-1],0)/h],wrap+np.array([max(v[0],0),max(-v[-1],0)])/h,0.,0.]; full=np.r_[face+[max(-v[0],0)/h,max(v[-1],0)/h],rule[2:]]
def stats(e): return np.sqrt((e.reshape(len(s['mu_shifts']),-1,len(d))**2).mean(1)).max(0)
for p in [x,rule,full]: print('rule',p,'rms',stats(err(p)))
for n in [4,6]:
 for held in [-1,0,1,2]:
  train=[i for i in range(len(d)) if i!=held]; pad=lambda p:np.r_[p,np.zeros(6-n)]
  fit=least_squares(lambda p:err(pad(p))[:,train].ravel(),x[:n],xtol=1e-13,gtol=1e-13,ftol=1e-13)
  print('fit',n,held,np.round(fit.x,7),'delta from simple rule',np.round(fit.x-rule[:n],7),'RMS',stats(err(pad(fit.x))))
