from explore import *
p,h,_=inputs[2]; X=m._features(di,di)[0]; r=theta[2]+X@theta[3:]; labs=di['lab_id']; obs=np.c_[di['strain_curves_obs']-di['strain_0'][:,None],di['instant_rate_obs']]; n,k=obs.shape

def temporal(ta):
 return np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]])

def setup(ta, variant, coeff):
 D=temporal(ta); base=h[:,None]*(D[:,:2]@theta[:2]); cols=[r[:,None]*D[:,b] for b in range(2)];
 if variant in ('free hold','free hold drift','free hold and intercept'):
  cols += [h[:,None]*D[:,b] for b in range(2)]
 if variant=='free hold and intercept': cols += [np.ones((n,1))*D[:,b] for b in range(2)]
 if variant in ('drift','free hold drift'): cols += [r[:,None]*D[:,2]]
 if variant=='history':
  age=np.cumsum(dt[::-1])[::-1]-dt; hist=p@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); cols += [hist[:,b:b+1]*D[:,b] for b in range(2)]
 if variant=='fixed': return base+np.stack(cols,-1)@np.array([.2,2.]),np.empty((n,k,0))
 return base,np.stack(cols,-1)

def fit(mask,variant,free):
 def project(ta):
  base,M=setup(ta,variant,None); A=M[mask].reshape(mask.sum()*k,M.shape[-1]); c=np.linalg.lstsq(A,(obs-base)[mask].ravel(),rcond=None)[0]; return base+M@c,c
 ta=tau if not free else np.exp(least_squares(lambda u: (project(np.exp(u))[0]-obs)[mask].ravel(),np.log(tau),bounds=(np.log([.2,1.5]),np.log([1.,8.]))).x)
 pred,c=project(ta); return pred,ta,c

for free in (False,True):
 for v in ('fixed','two projections','free hold','free hold and intercept','drift','free hold drift','history'):
  out=np.empty_like(obs); slices=[]
  for lab in np.unique(labs):
   mask=labs!=lab; out[~mask]=fit(mask,v,free)[0][~mask]; err=out[~mask]-obs[~mask]; slices.append([rms(err[:,:-1]),rms(err[:,-1])])
  err=out-obs; _,ta,c=fit(np.ones(n,dtype=bool),v,free); print(free,v,'OOF rms',np.round([rms(err[:,:-1]),rms(err[:,-1])],6),'worst lab',np.round(np.max(slices,axis=0),6),'max',np.round(np.max(abs(err),axis=0)[[np.argmax(np.max(abs(err[:,:-1]),axis=0)),-1]],6),'params',np.round(ta,6),np.round(c,6))
print('diag/holdout feature drift ranges', r.min(),r.max(),np.min(inputs[1][2]@theta[2:]),np.max(inputs[1][2]@theta[2:]))
