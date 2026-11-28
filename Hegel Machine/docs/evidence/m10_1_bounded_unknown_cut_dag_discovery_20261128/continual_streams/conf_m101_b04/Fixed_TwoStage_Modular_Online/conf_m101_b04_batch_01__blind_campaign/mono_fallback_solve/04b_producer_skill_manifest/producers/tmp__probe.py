import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; target=np.c_[y,t]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def lin(a):return np.c_[np.ones(len(a)),a]
def poly(z,d):return np.array([np.prod(z[:,c],axis=1) for c in combinations_with_replacement(range(z.shape[1]),d)]).T
p=s['panel_a']; u=p['u']; v=p['v']; a=lin(u); keep=np.arange(len(u));
for _ in range(8):
 ca=fit(a[keep],v[keep]); e=np.linalg.norm(v-a@ca,axis=1);keep=np.argsort(e)[:38]
print('A robust',ca,'residuals',np.sort(e).round(5));
p=s['panel_b'];cb=fit(lin(p['u']),p['v']); p=s['panel_c'];cc=[fit(lin(p['u']),p[k]) for k in ['v1','v2']];
zs={'X':x,'A':lin(x)@ca,'B':lin(x)@cb,'C1':lin(x)@cc[0],'C2':lin(x)@cc[1]};
for h in list(zs):
 for k,c in zip(['A','B','C1','C2'],[ca,cb,*cc]): zs[h+k]=lin(zs[h])@c
tr=np.arange(52); va=np.arange(52,72)
def check(a):
 c=fit(a[tr],target[tr]); err=np.abs(a[va]@c-target[va]); return [*np.round([err[:,0].mean(),err[:,1:].mean(),abs(a[tr]@c-target[tr]).mean(0)[0]],5)]
print('X poly',[check(np.c_[lin(x),*[poly(x,k) for k in range(2,d+1)]]) for d in range(1,5)])
for name,z in zs.items():
 q=np.c_[lin(x),poly(x,2)]; candidates=[]
 for trig in [np.sin,np.tanh,np.cos,lambda z:np.abs(z),lambda z:z**3]:
  for freq in [.7,1.3,2.,3.]:
   a=np.c_[q,trig(freq*z)]; candidates.append((check(a),trig.__name__,freq))
 print(name,sorted(candidates)[:2]);
print('y linear',fit(lin(x),y));print('traj linear',fit(lin(x),t)[:,[-1]]);print('resid corr',np.corrcoef((target-lin(x)@fit(lin(x),target)).T)[0]);
