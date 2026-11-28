import pickle,numpy as np
from scipy.optimize import root
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u,v=a['u'],a['v']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=52
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda z:np.c_[np.ones(len(z)),z]
def poly(z,k):return np.c_[aff(z),*[np.prod(z[:,idx],axis=1) for j in range(2,k+1) for idx in cr(range(z.shape[1]),j)]]
good=np.linalg.norm(v-u,axis=1)<4
b=fit(aff(s['panel_b']['u']),s['panel_b']['v']); print('B orthog', (b[1:]@b[1:].T).round(4));
coords={'raw':x}
for degree in (1,2,3,4,5):
 p=fit(poly(u[good],degree),v[good]); r=v[good]-poly(u[good],degree)@p; print('A poly',degree,np.sqrt(np.mean(r*r)),np.max(abs(r)))
 if degree<=3:
  roots=[root(lambda z:(poly(z[None,:],degree)@p)[0]-row,row) for row in x]; print('roots',max(np.linalg.norm(q.fun) for q in roots)); coords[f'inv{degree}']=np.asarray([q.x for q in roots]);coords[f'b_inv{degree}']=aff(coords[f'inv{degree}'])@b
for k,z in coords.items():
 for d in (1,2,3):
  f=poly(z,d); w=fit(f[:n],Y[:n]); err=np.mean(abs(f[n:]@w-Y[n:]),axis=0); tr=np.mean(abs(f[:n]@w-Y[:n]),axis=0); print(k,d,'val',np.round([err[0],err[1:].mean()],4),'fit',np.round([tr[0],tr[1:].mean()],4));
  if d==2 and k.startswith('b_inv'):print('coef y,tr',np.c_[w[:,0],w[:,-1]].round(3).T)
print('A coordinate fits')
for j in range(4):
 q=u[:,j]; p=u[:,(j+1)%4]; f=np.c_[np.ones(len(u)),q,p,q*p,q*q,p*p]; w=fit(f[good],v[good,j]); r=v[good,j]-f[good]@w; print(j,w.round(3),np.sqrt(np.mean(r*r)),np.corrcoef(np.c_[r,u[good],np.sin(2*u[good]),np.cos(2*u[good])].T)[0,1:].round(2))
