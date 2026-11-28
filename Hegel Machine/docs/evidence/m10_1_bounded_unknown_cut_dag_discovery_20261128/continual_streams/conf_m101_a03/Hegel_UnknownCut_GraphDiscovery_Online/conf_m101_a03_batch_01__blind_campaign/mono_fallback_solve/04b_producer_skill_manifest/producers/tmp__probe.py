import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; y=d['y']; t=d['traj']; x=d['x']
def a(x): return np.c_[np.ones(len(x)),x]
def fit(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
p=s['panel_a']; f=a(p['u']); r=np.linalg.norm(p['v']-f@fit(f,p['v']),axis=1); k=r<4*np.median(r); A=fit(f[k],p['v'][k]); u=(x-A[0])/np.diag(A[1:]); p=s['panel_b']; B=fit(a(p['u']),p['v']); z=a(u)@B
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=18).split(x)]
def score(f): return np.array([np.mean(abs(f[j]@fit(f[i],y[i])-y[j])) for i,j in splits])
def poly(x,n):
 idx=[j for k in range(1,n+1) for j in cwr(range(x.shape[1]),k)]; return a(np.column_stack([x[:,j].prod(1) for j in idx])),['1']+[''.join(map(str,j)) for j in idx]
for name,w in [('u',u),('z',z)]:
 for n in (1,2,3,4):
  f,nn=poly(w,n); c=fit(f,y); print(name,n,score(f).round(5), 'train',np.mean(abs(y-f@c)).round(5));
  if n<=3: print(sorted(zip(nn,c.round(4)),key=lambda p:-abs(p[1]))[:18]);
f=a(z); r=y-f@fit(f,y); print('state in z:',fit(f,t)[[0,1,2,3,4],:][:,[0,7]].round(5)); print('z ranges',z.min(0),z.max(0));
for name,w in [('u',u),('z',z)]:
 for mode in ('sin','cos','tanh','rat','abs'):
  for k in (.8,1.0,1.7,2.0,2.5,3.0):
   g={'sin':lambda x:np.sin(k*x),'cos':lambda x:np.cos(k*x),'tanh':lambda x:np.tanh(k*x),'rat':lambda x:x/(1+k*x*x),'abs':abs}[mode]; f0,_=poly(w,2); f=np.c_[f0,g(w)]; e=score(f); print(name,mode,k,e.round(4)) if e.max()<.07 else None
