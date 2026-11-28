exec(open('/tmp/test_solver.py').read().split("print('endpoint coefficients'")[0])
from scipy.optimize import least_squares
ix=np.arange(len(y)); o,g=m._calibration(s['panel_a']); z=(r['x']-o)/g
splits=[ix]+[np.random.default_rng(k).permutation(len(y)) for k in range(10)]+[np.argsort(np.max(abs(z),axis=1))]
slow_simple=np.c_[B,C]; fast_simple=np.c_[A,D]
for name in ('current','sparse','coupled','coupled_offset','coupled_fixed','coupled_fixed_offset'):
 scores=[]; pars=[]
 for p in splits:
  i,j=p[:56],p[56:]
  if name in ('current','sparse'):
   ds,df=(S,F) if name=='current' else (aff(slow_simple),F)
   ba,(cs,cf),q=m._fit_modes(ds[i],df[i],T,Y[i]); st=np.c_[ds[j]@cs,df[j]@cf]@ba.T
  else:
   fixed='fixed' in name; offset=name.endswith('offset')
   def design(index,q):
    b=m._time_basis(T,[.42,1.82] if fixed else q[:2]); slow,fast=b.T
    if offset:
     return np.column_stack(((slow_simple[index,None,:]*(slow+q[-1]*fast)[None,:,None]).reshape(-1,5),(fast_simple[index,None,:]*fast[None,:,None]).reshape(-1,5),np.tile(b,(len(index),1))))
    return np.column_stack(((slow_simple[index,None,:]*(slow+q[-1]*fast)[None,:,None]).reshape(-1,5),(fast_simple[index,None,:]*fast[None,:,None]).reshape(-1,5)))
   def profile(q):
    x=design(i,q); c=m._fit(x,Y[i].ravel()); return x@c-Y[i].ravel(),c
   q=least_squares(lambda q:profile(q)[0], [.1] if fixed else [.42,1.8,.1],bounds=([-.5],[.5]) if fixed else ([.08,1.,-.5],[.9,3.,.5])).x
   st=(design(j,q)@profile(q)[1]).reshape(len(j),len(T))
  e=abs(st-Y[j]); scores.append([e.mean(),e.mean(0).max(),e[:,-1].mean()]); pars.append(q)
 print(name,'ordered',np.round(scores[0],7),'mean',np.round(np.mean(scores,0),7),'worst',np.round(np.max(scores,0),7),'extreme',np.round(scores[-1],7),'parameter median',np.median(pars,axis=0))
