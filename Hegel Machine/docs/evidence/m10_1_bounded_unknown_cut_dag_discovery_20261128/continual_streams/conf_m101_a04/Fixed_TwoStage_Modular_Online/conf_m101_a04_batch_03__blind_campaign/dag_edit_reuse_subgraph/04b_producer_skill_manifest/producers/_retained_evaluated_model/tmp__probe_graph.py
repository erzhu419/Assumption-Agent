import runpy,pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
read=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['_readout_basis']
def affine(u): return np.c_[np.ones(len(u)),u]
def cyclic(u): return np.stack((np.ones_like(u),u,np.roll(u,-1,axis=1)*(u-1),u**3),axis=-1)
p=s['panel_a']; u,v=p['u'],p['v']; d=cyclic(u); w=np.ones(len(u))
for _ in range(25):
 q=np.sqrt(w); g=np.array([np.linalg.lstsq(d[:,j]*q[:,None],v[:,j]*q,rcond=None)[0] for j in range(4)])
 e=np.linalg.norm(np.einsum('njk,jk->nj',d,g)-v,axis=1); w=1/(1+(e/max(2*np.median(e),1e-8))**2)
def forward(z): return np.einsum('njk,jk->nj',cyclic(z),g)
def inv(x): return np.array([least_squares(lambda u:forward(u[None])[0]-r,(r-g[:,0])/g[:,1],gtol=1e-12).x for r in x])
p=s['panel_b']; b=np.linalg.lstsq(affine(p['u']),p['v'],rcond=None)[0]
def B(u): return affine(u)@b
p=s['panel_c']; u=p['u']
def fit(f,v,start):
 def design(u,a): return affine(f(u,a))
 def res(a):
  d=design(u,a); return (d@np.linalg.lstsq(d,v,rcond=None)[0]-v).ravel()
 a=least_squares(res,start).x; d=design(u,a); g=np.linalg.lstsq(d,v,rcond=None)[0]
 return lambda u:design(u,a)@g
C=fit(lambda u,a:u+a[1]*np.arcsinh(a[0]*u)+a[2]*u**3,p['v1'],[4.27,4.4,-.56]); D=fit(lambda u,a:u+a[1]*(np.cos(a[0]*u)-1)+a[2]*u*u,p['v2'],[2.52,-.061,.355]); T=fit(lambda u,a:u+a[1]*np.tanh(a[0]*u),p['v1'],[2.18,2.45])
x=s['train']['x']; X=inv(x); y=np.c_[s['train']['y'],s['train']['traj']]; n=len(x)
print('inverse max residual',abs(forward(X)-x).max(),'latent range',X.min(0),X.max(0))
def score(h):
 f=read(h); en=np.zeros((n,2));
 for j,(lam,k) in enumerate(((.0003,slice(0,1)),(.001,slice(1,None)))):
  # Exact leave-one-out for the frozen ridge readout.
  R=np.linalg.inv(f@f.T+lam*np.eye(n)); en[:,j]=np.mean(abs((R@y[:,k])/np.diag(R)[:,None]),axis=1)
 return en.mean(0),en.max(0)
features={}
for name,z in [('x',x),('A',X),('Bx',B(x)),('BA',B(X))]:
 c,d,t=C(z),D(z),T(z)
 for key,h in [('base',z),('C',c),('T',t),('D',d),('concat',np.c_[c,d]),('sum',c+d),('mean',(c+d)/2),('diff',c-d),('prod',c*d),('BC',B(c)),('BD',B(d)),('C+D+B',c+d+B(z)),('C+D+z',c+d+z),('C,D,z',np.c_[c,d,z])]: features[name+':'+key]=h
 for a in np.linspace(-2,3,101): features[f'{name}:mix {a:.2f}']=a*c+(1-a)*d
rank=[]
for key,h in features.items():
 mean,mx=score(h); rank.append((max(mean),key,*mean,*mx))
for r in sorted(rank)[:20]: print(r)
for r in rank:
 if 'mix' not in r[1]: print(r)
