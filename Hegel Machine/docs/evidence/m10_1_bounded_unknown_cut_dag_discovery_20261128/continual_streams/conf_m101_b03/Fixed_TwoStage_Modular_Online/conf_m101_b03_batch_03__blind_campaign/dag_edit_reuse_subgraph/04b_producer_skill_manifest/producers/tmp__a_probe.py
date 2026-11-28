import itertools as it,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; u=p['u']; v=p['v']; n=len(u)
A=np.c_[np.ones(n),u]; w=np.ones(n)
for _ in range(25):
 B=np.linalg.lstsq(A*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(A@B-v,axis=1); w=1/np.hypot(1,r/np.median(r))
keep=np.argsort(r)[:-3]; u=u[keep]; v=v[keep]; A=A[keep]
def probe(tag,g):
 D=np.c_[A,g]; B=np.linalg.lstsq(D,v,rcond=None)[0]; r=v-D@B; loo=r/(1-np.diag(D@np.linalg.pinv(D)))[:,None]; print(tag,np.round(np.sqrt(np.mean(r*r,axis=0)),6),np.round(np.sqrt(np.mean(loo*loo,axis=0)),6))
probe('affine',np.empty((len(u),0))); probe('sq',u*u); probe('cube',u**3); probe('quad',[[]]*len(u)) if False else None
W=np.stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)],axis=1)
probe('products',W); probe('quadratic',np.c_[u*u,W]); probe('triple',np.stack([np.prod(u[:,q],1) for q in it.combinations(range(4),3)],1))
for f in (.7,1,1.3,1.5,1.7,2,2.4,2.8,3,3.5):
 for name,g in [('sin',np.sin(f*u)),('cos',np.cos(f*u)),('tanh',np.tanh(f*u)),('sinprod',np.sin(f*W)),('cosprod',np.cos(f*W))]:probe((name,f),g)
