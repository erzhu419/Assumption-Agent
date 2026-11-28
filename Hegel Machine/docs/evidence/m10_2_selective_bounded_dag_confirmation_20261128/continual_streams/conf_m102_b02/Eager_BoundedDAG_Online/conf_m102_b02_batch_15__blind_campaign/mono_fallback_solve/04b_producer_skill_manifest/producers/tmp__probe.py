import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
def D(x):return np.c_[np.ones(len(x)),x]
def blocks(u):
 p=list(itertools.combinations(range(4),2)); a,b=np.array(p).T
 return dict(lin=u,sq=u*u,cub=u**3,adj=u*np.roll(u,-1,axis=1),pair=u[:,a]*u[:,b],triple=np.array([np.prod(u[:,i],axis=1) for i in itertools.combinations(range(4),3)]).T,sin=np.sin(2.15*u),cos=np.cos(2.15*u),tanh=np.tanh(2.15*u),soft=u/np.sqrt(1+(2.15*u)**2),rat=u/(1+(2.15*u)**2),abs=np.abs(u),exp=np.exp(-2.15*u*u))
for k in ['panel_a','panel_b','panel_c']:
 u=s[k]['u']; z=blocks(u)
 for key in s[k].keys()-{'u'}:
  v=s[k][key]; results=[]
  for j in range(1,5):
   for sub in itertools.combinations(z,j):
    f=D(np.column_stack([z[i] for i in sub])); n,m=f.shape
    if m>n-5:continue
    if k=='panel_a':
     keep=np.ones(n,bool)
     for _ in range(6):
      c=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; e=np.linalg.norm(f@c-v,axis=1); keep=np.argsort(e)[:40]
     ff,vv=f[keep],v[keep]
    else:ff,vv=f,v
    q=np.linalg.pinv(ff); e=(vv-ff@(q@vv))/(1-np.sum(ff*q.T,axis=1))[:,None]; results.append((np.mean(np.abs(e)),sub))
  print(k,key,sorted(results)[:12])
