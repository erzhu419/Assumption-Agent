import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
terms=[m for d in range(4) for m in itertools.combinations_with_replacement(range(4),d)]
def design(u,ts):return np.column_stack([np.prod(u[:,t],axis=1) for t in ts])
def fit(u,v,ts,rob=False):
 a=design(u,ts); w=np.ones(len(a)); v=np.asarray(v).reshape(len(a),-1)
 for _ in range(35 if rob else 1):
  c=np.linalg.lstsq(a*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(a@c-v,axis=1); w=np.sqrt(np.minimum(1,1.5*max(np.median(r),1e-10)/np.maximum(r,1e-10)))
 return c,r
u=s['panel_a']['u'];v=s['panel_a']['v']; print('telemetry',s['telemetry_vec']);
for k in range(4):
 for shift in (1,-1,2):
  j=(k+shift)%4; ts=[(),(k,),(j,),tuple(sorted((j,k))),(k,k,k)]; c,r=fit(u,v[:,k],ts,True); print('A',k,shift,c.ravel().round(4),np.quantile(r,[.5,.85,.9,1]).round(4))
for key in ['panel_b','panel_c']:
 for vk,v in s[key].items():
  if vk=='u':continue
  for d in [1,2]:
   ts=[t for t in terms if len(t)<=d]; c,r=fit(s[key]['u'],v,ts); print(key,vk,d,'rms',np.sqrt(np.mean(r*r)),[(t,row.round(3).tolist()) for t,row in zip(ts,c) if max(abs(row))>.015])
