exec(open('/tmp/probe14.py').read().split("print('T singular")[0])
from itertools import combinations_with_replacement
u,v=(s['panel_a'][k] for k in ('u','v')); A=D(u); b=ls(A,v)
for _ in range(8):
 r=np.linalg.norm(A@b-v,axis=1); mask=r<4*np.median(r);b=ls(A[mask],v[mask])
print('excluded',np.where(~mask)[0],r[~mask]); print('baseline',np.round(b,5))
blocks={str(ii):np.prod(u[:,ii],axis=1) for d in (2,3) for ii in combinations_with_replacement(range(4),d)}
for j in range(4):
 w=u[:,j];opp=u[:,(j+2)%4]; p=u[:,(j+1)%4];m=u[:,(j-1)%4];
 def fit(Z):
  c=ls(Z[mask],v[mask,j]); H=Z[mask]@np.linalg.pinv(Z[mask]);return np.std((Z[mask]@c-v[mask,j])/(1-np.diag(H))),c
 for name,terms in [('square',[w*w,opp*opp,w*opp,p*m,w**3]),('full2',list(blocks.values())[:10]),('full3',list(blocks.values())),('trig',[np.sin(2.15*w),np.cos(2.15*w),np.tanh(2.15*w),np.sin(2.15*opp),np.cos(2.15*opp),np.tanh(2.15*opp)]),('rat',[w/(1+2.15*w*w),opp/(1+2.15*opp*opp),p/(1+2.15*p*p),m/(1+2.15*m*m)])]:
  sc,c=fit(np.c_[A,*terms]);print(j,name,round(sc,6),np.round(c,3))
 feats=dict(blocks)
 for i in range(4):
  x=u[:,i]
  for k in (.22,1,2.15):
   for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('sat',lambda a:a/(1+a*a)),('abs',lambda a:a/(1+abs(a)))]:feats[f'{nm}{i}:{k}']=fn(k*x)
 Z=A.copy(); sel=[]
 for i in range(6):
  key=min(feats,key=lambda key:fit(np.c_[Z,feats[key]])[0]);Z=np.c_[Z,feats.pop(key)];sel.append(key);print('greedy',j,sel,round(fit(Z)[0],7), np.round(fit(Z)[1],4))
