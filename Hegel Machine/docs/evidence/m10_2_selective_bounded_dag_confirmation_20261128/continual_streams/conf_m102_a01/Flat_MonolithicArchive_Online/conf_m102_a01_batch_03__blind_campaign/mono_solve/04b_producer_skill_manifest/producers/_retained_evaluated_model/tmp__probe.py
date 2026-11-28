import sys,pickle,numpy as np
from itertools import combinations_with_replacement as comb
sys.path.insert(0,'/tmp/hegel_continual_archive');import conf_m102_a01_batch_00_solver as prior
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=52; np.set_printoptions(precision=5,suppress=True)
def poly(x,d):return np.stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in comb(range(x.shape[1]),k)],axis=1)
def fit(A,b):return np.linalg.lstsq(A,b,rcond=None)[0]
z,h=prior._coordinates(s,x); print('t,tele',s['eval_times'],s['telemetry_vec']); print('coords offset/basis',fit(poly(x,1),z),fit(poly(x,1),h))
for panel in ['panel_a','panel_b','panel_c']:
 p=s[panel];u=p['u']; print(panel)
 for k,v in p.items():
  if k=='u':continue
  for name,w in [('direct',v),('atanh',np.arctanh(np.clip(v,-.99999,.99999)))]:
   for d in [1,2,3]:
    A=poly(u,d); c=fit(A,w);print(k,name,d,np.sqrt(np.mean((A@c-w)**2)),np.sqrt(np.mean((A[-5:]@fit(A[:-5],w[:-5])-w[-5:])**2)))
  if panel=='panel_c':print('quadratic coefficients',fit(poly(u,2),v))
a=prior.solve_batch({**s,'train':{k:(v[:n] if k!='summary' else v) for k,v in s['train'].items()},'test':{'x':x[n:]}});print('archive y error',np.round(a['y_hat']-y[n:],4));
for w in [x,z,h]:
 for d in [1,2,3]:
  A=poly(w,d); b=np.c_[y,T]; c=fit(A[:n],b[:n]); print('polynomial',d,np.mean(abs(A[n:]@c-b[n:]),axis=0), 'magnitude residual y train',np.std((A[:n]@c-b[:n])[:,0]));
 print('T coeff linear',fit(poly(w,1),T));print('y coeff second',fit(poly(w,2),y))
