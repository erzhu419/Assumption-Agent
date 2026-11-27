import importlib.util,pickle,numpy as np
from scipy import sparse
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('ref','/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); Q=np.linalg.qr(s['Z_nuisance'],mode='reduced')[0]; axis=m._axis

def prediction(p,kind):
    ix=iter(p)
    def wrapped(k,v,h):
        A=axis(k,v,h).tolil()
        for i,j in ((0,1),(len(k)-1,len(k)-2)):
            d=next(ix); A[i,i]+=d/h
            if kind in ('edge','full'):
                t=next(ix)*k[i]/h**2; A[i,i]+=t; A[i,j]-=t
            if kind in ('wrap','full'):
                t=next(ix)*k[i]/h**2; A[i,i]+=t; A[i,len(k)-1-i]-=t
        return A.tocsc()
    m._axis=wrapped
    return m._forward(s)[0]@W

def resid(p,kind):
    r=prediction(p,kind)-Y
    return r-Q@(Q.T@r)
from itertools import product
for a in 'xy':
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(len(k)-1)
 print(a,'h',h,'k endpoints and neighbors',k[[0,1,-2,-1]],'endpoint velocities',v[[0,-1]])
 for i,j in [(0,1),(len(k)-1,len(k)-2)]:
  d=1 if i==0 else -1
  print('row',i,'adjacent diff/upwind correction',(k[j]/k[i]-1)/2,h*max(-d*v[i],0)/k[i], 'opposite diff/upwind', (k[0]+k[-1])/(2*k[i]), h*max(d*v[i],0)/k[i])

def seam(k,v,h,adj,drift,face):
 A=axis(k,v,h).tolil(); n=len(k)
 for i,j,d in [(0,1,1),(n-1,n-2,-1)]:
  l=n-1-i
  step=((k[i]+k[j])/2 if adj else k[i])/h**2
  across={'arithmetic':(k[i]+k[l])/2,'local':k[i], 'harmonic':2*k[i]*k[l]/(k[i]+k[l])}[face]/h**2
  if drift:
   step+=max(-d*v[i],0)/h
   across+=max(d*v[i],0)/h
  A[i,i]=step+across; A[i,j]=-step; A[i,l]=-across
 return A.tocsc()
for adj,drift,face in product([False,True],[False,True],['arithmetic','local','harmonic']):
 m._axis=lambda k,v,h:seam(k,v,h,adj,drift,face)
 r=m._forward(s)[0]@W-Y
 print(adj,drift,face,'pulse rms',np.sqrt(np.mean(r*r,axis=0)),'worst shift rms',np.sqrt(np.mean(r.reshape(len(s['mu_shifts']),-1,2)**2,axis=1)).max())
