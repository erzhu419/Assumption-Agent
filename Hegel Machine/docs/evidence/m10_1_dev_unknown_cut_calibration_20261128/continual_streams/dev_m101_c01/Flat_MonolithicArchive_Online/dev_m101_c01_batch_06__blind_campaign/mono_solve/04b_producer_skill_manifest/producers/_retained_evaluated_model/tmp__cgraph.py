exec(open('/tmp/explore.py').read().split('a=s[')[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
p=s['panel_a']; good=np.linalg.norm(p['v']-p['u'],axis=1)<2; A=np.array([ls(poly(p['u'][good,j:j+1],1),p['v'][good,j]) for j in range(4)]); a=(x-A[:,0])/A[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); b=poly(a,1)@B
p=s['panel_c']; u=p['u']; v=p['v1']; w=p['v2']
def c1(U,r): return np.c_[poly(U,1),np.tanh(U*r[:4]+r[4:])]
def c2(U,r): return np.c_[poly(U,1),np.cos(U*r),np.sin(U*r)]
def prof(fun,V,start,bounds,rows):
 def calc(r):
  F=fun(u,r); L=ls(F[rows],V[rows]);return F@L-V
 fit=least_squares(lambda r:calc(r)[rows].ravel(),start,bounds=bounds,max_nfev=200)
 return fit.x,ls(fun(u,fit.x)[rows],V[rows]),calc(fit.x)
for fun,V,start,lim in [(c1,v,[2.3]*4+[0]*4,([.4]*4+[-1]*4,[5]*4+[1]*4)),(c2,w,[1.2]*4,([.15]*4,[5]*4))]:
 for rows in [slice(18),slice(None)]:
  r,L,e=prof(fun,V,start,lim,rows);print('C',fun.__name__,r,'rms',np.sqrt(np.mean(e[:18]**2)),np.sqrt(np.mean(e[18:]**2)), 'coeff',L.T)
 if fun==c1: ca=lambda U,r=r,L=L:c1(U,r)@L
 else: cb=lambda U,r=r,L=L:c2(U,r)@L
rng=np.random.default_rng(512); splits=[np.arange(56,72)]+[k for _ in range(2) for k in np.array_split(rng.permutation(n),5)]
mode=np.linalg.svd(z,full_matrices=False)[2][0]; out=np.c_[y,z@mode]; inds=[(np.setdiff1d(np.arange(n),k),k) for k in splits]
def score(K):
 return np.array([np.mean(np.abs(K[np.ix_(iv,it)]@cho_solve(cho_factor(K[np.ix_(it,it)]),out[it])-out[iv]),axis=0) for it,iv in inds])
P=poly(a,2); K=P@P.T; eye=np.eye(n)
for nm,X in [('raw',x),('a',a),('b',b),('C1a',ca(a)),('C1b',ca(b)),('C2a',cb(a)),('C2b',cb(b)),('C1C2a',np.c_[ca(a),cb(a)]),('C1C2b',np.c_[ca(b),cb(b)])]:
 d=cdist(X,X,'sqeuclidean'); configs=[]
 for label,H in [('poly',poly(X,2)),('x cross',np.c_[X,X[:,:,None].__mul__(a[:,None,:]).reshape(n,-1)]),('lin',poly(X,1))]:
  G=H@H.T
  for k in [.1,1,10]:
   E=score(K+k*G+.001*eye);configs.append((E.mean(0)+.4*E.max(0),E[0],E.mean(0),E.max(0),(label,k)))
 for l in [1,1.5,2,3]:
  for m in [.1,.5,1.]:
   r=np.sqrt(5*(d/l**2+m*cdist(a,a,'sqeuclidean'))); G=(1+r+r*r/3)*np.exp(-r)
   for k in [.05,.5,5]:
    E=score(K+k*G+.0003*eye);configs.append((E.mean(0)+.4*E.max(0),E[0],E.mean(0),E.max(0),('matern',l,m,k)))
 print(nm,*[min(configs,key=lambda e:e[0][i])[1:] for i in range(2)])
