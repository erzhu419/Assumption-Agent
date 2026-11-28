import sys,pickle
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations
sys.path.insert(0,'/holdout_task')
from batch_solver import _coordinates,_models,_affine
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x,z=_coordinates(s,t['x']); y=np.c_[t['y'],t['traj']]; n=len(x)

def additions(x,z,y):
 n=len(y); B=_affine(z); h,r=B[:n],B[n:]; d=lambda a:cdist(a/a[:n].std(0),a/a[:n].std(0),'sqeuclidean')/a.shape[1]; dx,dz=d(x),d(z); sine=np.sin(2.8*x); ds=d(sine); dp=[d(x[:,ij]) for ij in list(combinations(range(4),2))]; pair=slice(None)
 for w in [.24,.34,.48,.68,.96,1.36,1.92]:
  c=2*w*w; J=np.exp(-(dx+dz)/(2*c)); Q=np.exp(-(dx+ds)/(2*c)); P=np.exp(-np.array(dp)/c).mean(0)
  for label,K in [('xz75',np.exp(-(.25*dx+.75*dz)/c)),('xz25',np.exp(-(.75*dx+.25*dz)/c)),('xs',Q),('xsz',np.exp(-(.5*dx+.25*dz+.25*ds)/c)),('ax',P*J),('pairplus',P+.4*J),('poly', (1+(x/x[:n].std(0))@(x/x[:n].std(0)).T/4)**3+J)]:
   for lam in [.001,.005,.02,.08,.32,1.28]:
    F=cho_factor(K[:n,:n]+lam*np.eye(n),check_finite=False); A=cho_solve(F,np.eye(n),check_finite=False); G=A@h; F2=np.linalg.solve(h.T@G,G.T); P0=A-G@F2; C=P0@y; E=abs(C/np.diag(P0)[:,None]); score=np.array([E[:,0].mean(),E[:,1:].mean(),E[:,-1].mean()]); yield label,score,K[n:,:n]@C+r@(F2@y)

def assess(ii):
 a,b=ii[:56],ii[56:]; old=list(_models(x[ii],z[ii],y[a])); new=list(additions(x[ii],z[ii],y[a])); vals={}
 for label,models in [('old',old),('all',old+new),('zx',[q for q in old+new if q[0] in ('joint','xz25','xz75')]),('xs',[q for q in new if q[0] in ('xs','xsz')]),('match',[q for q in old+new if q[0] in ('joint','pair','xz25','xz75','xs','xsz')])]:
  score=np.array([q[1] for q in models]); pred=np.array([q[2] for q in models]); obj=score@[1,1.4,.2]; best=obj.argmin()
  for temp in [0,.0015,.003,.006,.013]:
   if temp: w=np.exp(-(obj-obj.min())/temp); p=np.einsum('j,jik->ik',w/w.sum(),pred)
   else: p=pred[best]
   e=abs(p-y[b]); vals[(label,temp)]=[e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
 return vals
rng=[np.arange(n)]+[np.random.default_rng(k).permutation(n) for k in range(16)]; radial=[np.argsort(np.linalg.norm(q/q.std(0),axis=1)) for q in (x,z)]; axes=[np.argsort(abs(q[:,i]-np.mean(q[:,i]))) for q in (x,z) for i in range(4)]; result=[assess(ii) for ii in rng+radial+axes]; m=len(rng)
for k in result[0]:
 e=np.array([r[k] for r in result]);print(k,np.round([e[0],e[:m].mean(0),e[:m].max(0),e[m:].mean(0),e[m:].max(0)],5).tolist())
