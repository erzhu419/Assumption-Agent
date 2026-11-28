import pickle,numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
poly=lambda x,d:PolynomialFeatures(d).fit_transform(x)
p=s['panel_a']; u,v=p['u'],p['v']; m=np.linalg.norm(u-v,axis=1)<1
A=np.array([ls(poly(u[m,i:i+1],1),v[m,i]) for i in range(4)]); af=lambda x:x*A[:,1]+A[:,0]; ai=lambda x:(x-A[:,0])/A[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); bf=lambda x:poly(x,1)@B
p=s['panel_c']; u=p['u']; C=[]
for key in ['v1','v2']:
 v=p[key]; print(key)
 for d in [1,2,3]:
  F=poly(u,d); es=[]
  for a,b in KFold(6,shuffle=True,random_state=11).split(u):es.extend(abs(F[b]@ls(F[a],v[a])-v[b]).ravel())
  coef=ls(F,v); print(d,'train',np.mean(abs(F@coef-v)),'CV',np.mean(es));
  if d==2: print(np.round(coef,4)); C.append(coef)
for key in ['v1','v2']:
 v=p[key]; print(key,'marginal range',v.min(0),v.max(0))

def evaluate(F):
 o=[]
 for a,b in [(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=12).split(x)]:
  pred=F[b]@ls(F[a],np.c_[y[a],t[a]]); o.append(np.mean(abs(pred-np.c_[y[b],t[b]]),axis=0))
 o=np.array(o); return tuple(np.round([o[0,0],o[1:,0].mean(),o[1:,0].max(),o[0,1:].mean()],5))
res=[]
for nm,q in [('raw',x),('A+',af(x)),('A-',ai(x)),('BA+',bf(af(x))),('BA-',bf(ai(x))),('Braw',bf(x)),('A-B',ai(bf(x))),('A+B',af(bf(x)))]:
 for d in [1,2,3]:
  T=poly(q,d); res.append((evaluate(T),nm+str(d)))
 f,g=[poly(q,2)@c for c in C]
 for name,z in [('C1',f),('C2',g),('fullC',np.c_[f,g]),('C1C2prod',f*g),('C1+C2',f+g),('C1-C2',f-g),('C2-C1sq',g-f*f),('tanhC1',np.tanh(f)),('tanhC2',np.tanh(g))]:
  for d in [1,2]:
   T=poly(z,d); res.append((evaluate(T),nm+name+str(d))); res.append((evaluate(np.c_[poly(q,2),T[:,1:]]),nm+'Q+'+name+str(d)))
print('best first split');print(*sorted(res)[:26],sep='\n');print('best worst folds');print(*sorted(res,key=lambda a:a[0][2])[:20],sep='\n')
