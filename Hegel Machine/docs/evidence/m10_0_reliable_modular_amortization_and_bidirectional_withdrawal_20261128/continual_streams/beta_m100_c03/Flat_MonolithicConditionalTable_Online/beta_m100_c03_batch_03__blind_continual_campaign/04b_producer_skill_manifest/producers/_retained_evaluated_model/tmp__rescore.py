import compare_chamber as q
from rosette import _calibrate
import itertools,numpy as np
q.y=_calibrate(q.s,q.s['y_obs_raw'],report=True)
print('study keys',list(q.s),'shifts',q.s['mu_shifts'])
print('obs/nuisance/source',q.y.shape,q.Z.shape,q.s['source_modes'].shape)
print('calibrated range',np.min(q.y),np.max(q.y))
print('robins',q.s['robin_x'],q.s['robin_y'])
kinds=[0,.25,.5,1,2,4,'periodic','absorbing']; i=np.arange(len(q.y)); Ax={k:q.axis('x',k) for k in kinds}; Ay={k:q.axis('y',k) for k in kinds}
for ridge in [q.s['source_ridge'],1e-8]:
 results=[]
 for a,b in itertools.product(kinds,repeat=2):
  S,_=q.m._responses(q.s,(Ax[a],Ay[b])); e,w=q.evaluate(S,i,ridge); fs=[]
  for fold in range(6):
   test=(i//4)%6==fold; err,_=q.evaluate(S,~test,ridge); fs.append(np.sqrt(np.mean(err[test]**2)))
  results.append((float(np.mean(fs)),max(fs),np.std(e),a,b,w.round(3)))
 print('ridge',ridge,'best',*sorted(results,key=lambda x:x[0])[:8],sep='\n'); print('core',*[r for r in results if r[3] in [0,1,'periodic'] and r[4] in [0,1,'periodic']],sep='\n')
