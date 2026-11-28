import numpy as np,pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
fun={'tanh':np.tanh,'asinh':np.arcsinh,'sin':np.sin,'cos':np.cos,'soft':lambda a:a/np.sqrt(1+a*a),'rat':lambda a:1/(1+a*a),'atan':np.arctan,'log':lambda a:np.log1p(a*a),'sq':np.square,'exp':lambda a:np.exp(-a*a)}
def D(*ar):return np.column_stack((np.ones(len(ar[0])),*ar))
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
for k in ['a','b','c']:
 d=s['panel_'+k]; u=d['u'];
 for key,v in d.items():
  if key=='u':continue
  keep=np.ones(len(u),bool)
  if k=='a':keep=np.max(abs(u-v),1)<3
  u0,v0=u[keep],v[keep]; cand=[]
  for f,fn in fun.items():
   for q in [.22,.5,1,1.5,2,2.15,2.5,3,3.5,4,4.3,5,6]:
    for mode,ww in [('u',u0),('u2',u0**2),('diff',u0-np.roll(u0,2,axis=1)),('opp',u0+np.roll(u0,2,axis=1))]:
     for a in (D(fn(q*ww)),D(u0,fn(q*ww))):
      c=ls(a,v0);err=np.sqrt(np.mean((a@c-v0)**2));cand.append((err,f,q,mode,a.shape[1],c))
  print(k,key,[(round(t[0],6),*t[1:5]) for t in sorted(cand,key=lambda t:t[0])[:10]]); print(min(cand,key=lambda t:t[0])[-1])
