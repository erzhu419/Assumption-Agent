import pickle,runpy,numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
from math import factorial
p='/holdout_task/'; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); b=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']({'node_00':a})
u,z=a['u_tr'],b['z_tr']; y=s['train']['y']; n=len(y); I=np.arange(n); L=np.c_[np.ones(n),u]; Gram=u@u.T; dist=cdist(u,u,'sqeuclidean')
splits=[np.arange(52,72)]+list(np.split(np.random.default_rng(982).permutation(n),6))+[np.argsort(np.linalg.norm(u,axis=1))[-12:]]
def evaluate(K, lam, yy=y, dec=lambda v:v):
 out=[]
 for te in splits:
  tr=np.setdiff1d(I,te); Z=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),np.c_[L[tr],yy[tr]]); c=np.linalg.solve(L[tr].T@Z[:,:5],L[tr].T@Z[:,5]); yp=L[te]@c+K[np.ix_(te,tr)]@(Z[:,5]-Z[:,:5]@c); out.append(np.mean(abs(dec(yp)-y[te])))
 return np.array([out[0],np.mean(out[1:7]),max(out[1:7]),out[-1]])
table=[]
for kind in ('poly','rbf','matern','mixed'):
 for length in ((.3,.6,1,1.7,3,5) if kind!='poly' else (.1,.25,.5,1,2)):
  for degree in ((3,4,5,7) if kind=='poly' else (0,)):
   r=np.sqrt(5*dist)/length
   if kind=='poly': K=(1+length*Gram)**degree
   if kind=='rbf': K=np.exp(-dist/(2*length**2))
   if kind=='matern': K=(1+r+r*r/3)*np.exp(-r)
   if kind=='mixed':
    d=np.c_[u,z]; sep=sum(np.exp(-(d[:,j,None]-d[None,:,j])**2/(2*length**2)) for j in range(8)); K=sep*(1+Gram)+.1*(1+.5*Gram)**3
   K=K/np.mean(np.diag(K))
   for lam in (1e-5,1e-4,.001,.003,.01,.03):
    m=evaluate(K,lam); table.append((kind,length,degree,lam,m))
for kind in ('poly','rbf','matern','mixed'):
 print(kind,[(v[:4],v[4].round(5)) for v in sorted([v for v in table if v[0]==kind],key=lambda v: max(v[4][:1].max(),v[4][2])+.25*v[4][3])[:6]])
X=np.c_[np.ones(n),*[np.prod(u[:,j],axis=1) for d in (1,2,3) for j in combinations_with_replacement(range(4),d)]]; beta=np.linalg.lstsq(X,y,rcond=None)[0]; res=y-X@beta
print('cubic RMSE',np.std(res),'target range',y.min(),y.max())
for d in (1,2,3):
 start=(1,5,15)[d-1]; terms=list(combinations_with_replacement(range(4),d)); print('degree',d,sorted(zip(terms,np.round(beta[start:start+len(terms)],5)),key=lambda v:-abs(v[1])))
print('relations of y with trajectory:',np.corrcoef(np.c_[y,s['train']['traj'][:,-1],u,z].T)[0]); print('panel c feature residuals')
c=s['panel_c']; bu=s['panel_b']['u']; bv=s['panel_b']['v']; B=np.linalg.lstsq(np.c_[np.ones(len(bu)),bu],bv,rcond=None)[0]; cu=c['u']; cz=np.c_[np.ones(len(cu)),cu]@B; A=np.c_[np.ones(len(cu)),cu]; Q=np.linalg.qr(A)[0]
for k in ('v1','v2'):
 R=c[k]-Q@(Q.T@c[k]); print(k,'resid sv',np.linalg.svd(R,compute_uv=False),'affine map',np.linalg.lstsq(A,c[k],rcond=None)[0].round(3));
 for vs,nm in ((cu,'u'),(cz,'z')):
  P=np.c_[*[vs[:,i]*vs[:,j] for i,j in combinations_with_replacement(range(4),2)]]; F=P-Q@(Q.T@P); print(nm,'quad residual max per channel',np.max(abs(R-F@np.linalg.lstsq(F,R,rcond=None)[0]),axis=0).round(5))
