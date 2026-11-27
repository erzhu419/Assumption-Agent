from chamber_probe import *
from chamber_probe import _chamber,_features
from scipy.optimize import least_squares
mem,mr=_chamber(tau,th,ha,drift*0,t)
E=cy-mem; R=ry-mr
print('slope through origin using measured rate',np.round(R@E/(R@R),5))
print('diagnostic feature alternatives:')
for name,xx in [('current',xq),('diagnostic only',_features(q)),('train + diag',_features(a,q)[-len(q['lab_id']):]),('test + diag',_features(b,q)[-len(q['lab_id']):])]:
 D=th[2]+xx@th[3:]; print(name,'rate rmse',np.sqrt(np.mean((D-R)**2)), 'lab rmse', [(int(l),round(np.sqrt(np.mean((D[q['lab_id']==l]-R[q['lab_id']==l])**2)),5)) for l in np.unique(q['lab_id'])]); print('rate residual',np.round(R-D,4))
print('time kernels profiled against measured rate:')
for name,fn in [('log',lambda k:np.log1p(k*t)/k),('exp',lambda k:-np.expm1(-k*t)/k),('rational',lambda k:t/(1+k*t)),('sqrt',lambda k:2*np.expm1(.5*np.log1p(k*t))/k),('quad',lambda k:t-k*t*t),('blend',lambda k:(1-k)*t+k*np.log1p(t))]:
 fit=least_squares(lambda p:(E-R[:,None]*fn(p[0])).ravel(),[.3],bounds=(1e-6,3.)); rs=E-R[:,None]*fn(fit.x[0]); print(name,fit.x,'rmse',np.sqrt(np.mean(rs**2)), 'horizons', np.round(np.sqrt(np.mean(rs**2,axis=0)),4))
print('apparatus log keys and compact endpoints:')
import json
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print([(k,(len(v),str(v[0])[:80],str(v[-1])[:80]) if isinstance(v,list) and v else str(v)[:320]) for k,v in row.items()])
