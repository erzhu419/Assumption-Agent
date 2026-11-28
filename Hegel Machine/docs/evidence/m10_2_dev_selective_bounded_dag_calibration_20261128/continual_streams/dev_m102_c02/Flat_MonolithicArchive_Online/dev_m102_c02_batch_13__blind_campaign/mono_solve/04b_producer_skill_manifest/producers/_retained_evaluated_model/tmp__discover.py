import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]
u,v=(s['panel_a'][k] for k in ('u','v')); ok=np.linalg.norm(v,axis=1)<5;u,v=u[ok],v[ok]; print('telemetry',s['telemetry_vec'],'times',s['eval_times'])
for j in range(4):
 a,b,c,d=np.roll(u,-j,axis=1).T; H=np.array([np.prod(np.array([a,b,c,d]).T**i,axis=1) for i in itertools.product(range(4),repeat=4) if sum(i)<=3]).T; names=[i for i in itertools.product(range(4),repeat=4) if sum(i)<=3]; resid=v[:,j].copy(); J=[0];
 for _ in range(10):
  r=v[:,j]-H[:,J]@fit(H[:,J],v[:,j]); hh=H-H[:,J]@fit(H[:,J],H); sc=abs(r@hh)/np.maximum(1e-9,np.linalg.norm(hh,axis=0));sc[J]=-1;J.append(np.argmax(sc)); print(j,len(J),names[J[-1]],np.std(v[:,j]-H[:,J]@fit(H[:,J],v[:,j])))
 print(list(zip([names[i] for i in J],fit(H[:,J],v[:,j]).round(5))))
