import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; np.set_printoptions(precision=5,suppress=True,linewidth=120)
def aff(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,ix],axis=1) for k in range(1,d+1) for ix in combinations_with_replacement(range(x.shape[1]),k)])
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
maps={}
for p in ['a','b','c']:
 a=s['panel_'+p]; u=a['u']; f=aff(u)
 for k,v in a.items():
  if k=='u':continue
  m=ls(f,v)
  if p=='a':
   for _ in range(15):
    ix=np.argsort(np.linalg.norm(f@m-v,axis=1))[:40];m=ls(f[ix],v[ix])
  print(p,k,'map',m,'row errors',np.sort(np.linalg.norm(f@m-v,axis=1)))
  if p!='a':print('quad RMS',np.sqrt(np.mean((poly(u,2)@ls(poly(u,2),v)-v)**2,axis=0)))
  maps[p+k]=m
u=(x-maps['av'][0])/maps['av'][1:].diagonal(); aa=ls(maps['av'][1:].T,(x-maps['av'][0]).T).T
print('SVD', np.linalg.svd(z,full_matrices=0)[1]);print('temporal bases',np.linalg.svd(z,full_matrices=0)[2][:4]); print('linear coefficients on raw',ls(aff(x),np.c_[y,z]).T);print('mean temporal',z.mean(axis=0));print('cor y/z',np.corrcoef(y,z.T)[0,1:]);
def report(n,a,b):
 pred=a[52:]@ls(a[:52],b[:52]); print(n,np.mean(abs(pred-b[52:]),axis=0))
for name,w in [('x',x),('u',u),('aa',aa)]+[(p+n,aff(h)@m) for p,h in [('x',x),('u',u),('a',aa)] for n,m in maps.items() if n!='av']:
 for f,F in [('id',lambda a:a),('rat',lambda a:a/np.sqrt(1+.22*a*a)),('rat2',lambda a:a/(1+.22*a*a)),('tanh',lambda a:np.tanh(a)),('tanh125',lambda a:np.tanh(1.25*a)),('sin',lambda a:np.sin(a))]:
  for d in [1,2,3]:
   a=poly(F(w),d); b=np.c_[y,z]; pred=a[52:]@ls(a[:52],b[:52]); e=np.mean(abs(pred-b[52:]),axis=0);print('cv',name,f,d,e[0],e[1:].mean(),e[-1])
