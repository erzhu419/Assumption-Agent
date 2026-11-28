import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; n=len(y)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(D,y): return np.linalg.lstsq(D,y,rcond=None)[0]
def poly(x,k=2):
 return np.column_stack([np.prod(x[:,ii],axis=1) for j in range(k+1) for ii in itertools.combinations_with_replacement(range(x.shape[1]),j)])
u,v=[s['panel_a'][k] for k in ('u','v')]; z=u.copy()
for j in range(4):
 d=aff(u[:,j:j+1]); w=ls(d,v[:,j]);
 for _ in range(8):
  ids=np.argsort(abs(d@w-v[:,j]))[:36]; w=ls(d[ids],v[ids,j]);
 z[:,j]=u[:,j]*w[1]+w[0]
g=np.std(z,axis=0)/np.std(u,axis=0); a=np.mean(z-u*g,axis=0); x=(tr['x']-a)/g
p=s['panel_b']; W=ls(aff(p['u']),p['v']); b=aff(x)@W
p=s['panel_c']; u=p['u'];
models={'x':x,'b':b}
for ky in ('v1','v2'):
 v=p[ky]; print('panel',ky)
 for kind,fn in {'q':lambda z:poly(z), 'sin':lambda z:np.c_[aff(z),np.sin(z),np.cos(z)], 'sin2':lambda z:np.c_[aff(z),np.sin(2*z),np.cos(2*z)], 'tanh':lambda z:np.c_[aff(z),np.tanh(z)], 'cubic':lambda z:np.c_[aff(z),z**2,z**3]}.items():
  D=fn(u); w=ls(D,v); h=np.sum(D*np.linalg.pinv(D).T,axis=1); print(kind,np.sqrt(np.mean(((D@w-v)/(1-h[:,None]))**2)))
  models[ky+kind+'/x']=fn(x)@w; models[ky+kind+'/b']=fn(b)@w
scores=[]; ix=np.arange(52); jx=np.arange(52,72); ix2=np.arange(16,72); jx2=np.arange(16)
def add(name,D):
 scores.append((np.mean(abs(D[jx]@ls(D[ix],y[ix])-y[jx])),np.mean(abs(D[jx2]@ls(D[ix2],y[ix2])-y[jx2])),np.mean(abs(D[jx]@ls(D[ix],T[ix])-T[jx])),name))
for name,z in models.items():
 for scale in (.3, .6, 1., 1.6, 2.5, 4.):
  for fn in (np.arcsinh,np.arctan,np.tanh,np.sin):
   h=fn(z*scale); add((name,fn.__name__,scale),poly(h)); add((name,fn.__name__,scale,'+raw'),np.c_[poly(h),z])
 for k in (1,2,3): add((name,'poly',k),poly(z,k))
for i in sorted(scores)[:24]: print(i)
print('best traj', sorted(scores,key=lambda a:a[2])[:10])
