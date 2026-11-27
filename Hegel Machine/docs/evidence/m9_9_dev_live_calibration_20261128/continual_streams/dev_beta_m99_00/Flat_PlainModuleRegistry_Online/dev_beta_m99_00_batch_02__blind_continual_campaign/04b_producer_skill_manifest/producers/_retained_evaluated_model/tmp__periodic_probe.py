import sys,json,pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
path='/tmp/wrap_probe.py'
exec(open(path).read().split('from itertools import product')[0])
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
ps=item['campaign_diagnostics']; modes=np.stack([np.tensordot(p['w_pulse'],s['source_modes'],axes=1) for p in ps]); y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in ps])
def coeff(inface, seam, adv):
    inward=[]; across=[]
    for axis in 'xy':
        k=s['kappa_'+axis]; v=s['vel_'+axis]; h=s['L'+axis]/(len(k)-1)
        e=k[[0,-1]]; near=k[[1,-2]]; t=v[[0,-1]]*h*np.array([1,-1]);
        inward.extend(((e+near)/2 if inface else e)/e + (np.maximum(-t,0)/e if adv else 0)); across.extend((e.mean() if seam else e)/e + (np.maximum(t,0)/e if adv else 0))
        if inface and seam and adv: print(axis,'endpoint k',e,'near',near,'v*h',t)
    return np.r_[np.zeros(4),inward,across]
for i in (False,True):
 for j in (False,True):
  for k in (False,True):
   p=coeff(i,j,k); e=forward(s,p,modes)-y; print('avg inward, avg seam, upwind',i,j,k,'params',p[4:].round(7),'pulse RMSE',np.sqrt(np.mean(e*e,axis=0)))
p=coeff(True,True,True)
for hold in range(len(ps)):
    keep=np.arange(len(ps))!=hold
    fit=least_squares(lambda x:(forward(s,np.r_[np.zeros(4),x],modes[:, :, :][keep])-y[:,keep]).ravel(),p[4:],bounds=(0,10),gtol=1e-13,ftol=1e-13,xtol=1e-13)
    e=forward(s,np.r_[np.zeros(4),fit.x],modes[hold:hold+1])[:,0]-y[:,hold]
    print('holdout',hold,'free conservative seam fit',fit.x.round(7),'RMSE',np.sqrt(np.mean(e*e)))
