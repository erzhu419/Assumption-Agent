import numpy as np,pickle,itertools as it
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
x=s['train']['x'];Y=np.c_[s['train']['y'],s['train']['traj']]
def aff(u):return np.c_[np.ones(len(u)),u]
def poly(u,k):return np.c_[np.ones(len(u)),*[np.prod(u[:,j],axis=1) for d in range(1,k+1) for j in it.combinations_with_replacement(range(4),d)]]
def fit(u,v):return lstsq(u,v)[0]
if __name__=='__main__':
 for name in ('panel_a','panel_b','panel_c'):
  p=s[name];u=p['u']
  for key,v in p.items():
   if key=='u':continue
   keep=np.arange(len(u))
   if name=='panel_a':
    e=np.linalg.norm(v-aff(u)@fit(aff(u),v),axis=1);keep=np.argsort(e)[:-3]
   a=u[keep];b=v[keep]; f=aff(a); w=fit(f,b); e=b-f@w
   print(name,key,'affine',w.T,'residual RMS',np.sqrt(np.mean(e**2,0)))
   for d in (2,3):
    f=poly(a,d); w=fit(f,b); e=b-f@w; print('degree',d,'rms',np.sqrt(np.mean(e**2,0)))
    if d==2:print(w[5:].T,'order',list(it.combinations_with_replacement(range(4),2)))
    if d==3 and name=='panel_a':
     names=['1']+[''.join(map(str,j)) for k in range(1,d+1) for j in it.combinations_with_replacement(range(4),k)];print([[(n,round(c,4)) for n,c in zip(names,r) if abs(c)>.01] for r in w.T])
   print('panel x range',a.min(0),a.max(0))
 p=s['panel_a'];u=p['u'];v=p['v'];e=np.linalg.norm(v-aff(u)@fit(aff(u),v),axis=1);keep=np.argsort(e)[:-3];w=fit(aff(u[keep]),v[keep]);z=(x-w[0])@np.linalg.inv(w[1:]);
 for label,u in [('raw',x),('invAffine',z)]:
  for d in (1,2,3):
   f=poly(u,d);res=f[52:]@fit(f[:52],Y[:52])-Y[52:];print(label,d,np.mean(abs(res[:,0])),np.mean(abs(res[:,1:])))
