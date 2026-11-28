exec(open('/tmp/stress_rational.py').read().split("print('public fields'")[0])
from functools import lru_cache
@lru_cache(None)
def pred(a,i,j,kind='rat'):
 return candidate(np.array(i),np.array(j),a,kind)[0]
# In addition to the existing splits, hold out each calibrated coordinate tail,
# and the highest-leverage inputs as measured from unlabeled input geometry.
old=len(splits)
l=bs._aff(z); h=np.sum(l*np.linalg.solve(l.T@l,l.T).T,axis=1)
for k,u in enumerate([*z.T,*(-z.T),h]):
 ix=np.argsort(u,kind='stable'); splits.append((f'coordinate{k}',ix[:-12],ix[-12:]))
for k in range(12):
 ix=rng.permutation(ids); splits.append((f'newrand{k}',ix[:52],ix[52:]))
policies=[('baseline',[(0,1,'rat')]),('mild',[(.004,1,'rat')]),('moderate',[(.006,1,'rat')]),('mix-shrink',[(.002,.5,'rat'),(.01,.5,'rat')]),('mix-shape',[(.004,.75,'rat'),(.004,.25,'root')])]
for name,parts in policies:
 e=[]
 for _,i,j in splits:
  p=sum(w*pred(a,tuple(i),tuple(j),kind) for a,w,kind in parts)
  e.append(np.mean(abs(p-y[j])))
 e=np.array(e); k=e.argmax()
 print(name,'gate',round(e[0],6),'random mean/max',np.round([np.mean(np.r_[e[1:9],e[old+9:]]),max(np.r_[e[1:9],e[old+9:]])],6),'old tails',e[9:old].round(6),'new tails',e[old:old+9].round(6),'worst',splits[k][0],round(e[k],6))
