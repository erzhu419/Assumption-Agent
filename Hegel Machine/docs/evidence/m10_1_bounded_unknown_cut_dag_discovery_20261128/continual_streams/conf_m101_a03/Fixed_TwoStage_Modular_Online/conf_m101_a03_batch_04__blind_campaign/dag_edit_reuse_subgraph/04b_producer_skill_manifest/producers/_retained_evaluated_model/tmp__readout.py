import pickle,numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement as cr,combinations as co
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=np.load('/tmp/panelcoords.npz')['z'][:len(s['train']['x'])]; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(z); splits=[(np.arange(n-20),np.arange(n-20,n))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(32).permutation(n),4)]
def std(X):return X/np.maximum(X.std(0),.1)
def cross(u):return np.column_stack([u[:,i]*u[:,j] for i,j in co(range(4),2)])
def poly(u,d):return np.column_stack([np.prod(u[:,i],axis=1) for j in range(1,d+1) for i in cr(range(4),j)])
F={}; candidates={}
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[panel]; u,v=p['u'],p[key]; fits=[]
 for label,make,lo,hi in [('tanh',lambda u,a:np.c_[u,np.tanh(a*u)],.4,3),('tanh_cross',lambda u,a:np.c_[u,np.tanh(a*u),cross(u)],.4,3),('atan_cross',lambda u,a:np.c_[u,np.arctan(a*u),cross(u)],.4,4),('tanh_cubic',lambda u,a:np.c_[u,np.tanh(a*u),u**3],.4,3),('rational',lambda u,a:np.c_[u,u/(1+a*u*u),cross(u)],.1,5),('quadratic',lambda u,a:np.c_[u,u*u],0,1),('square_tanh',lambda u,a:np.c_[u,np.tanh(a*u*u),u*u],.1,4),('square_sin',lambda u,a:np.c_[u,np.sin(a*u*u),u*u],.1,4),('quad3',lambda u,a:np.c_[u,u*u,u**3],0,1)]:
  def fit(a):
   A=np.c_[np.ones(len(u)),make(u,a)]; B=np.linalg.lstsq(A,v,rcond=None)[0]; e=A@B-v; h=np.sum(A*np.linalg.pinv(A).T,axis=1); return np.mean((e/(1-h[:,None]))**2)**.5,B
  r=minimize_scalar(lambda a:fit(a)[0],bounds=(lo,hi),method='bounded'); err,B=fit(r.x); pred=np.c_[np.ones(n),make(z,r.x)]@B; fits.append((err,label,r.x,pred,B))
 fits.sort(key=lambda v:v[0]); print('PANEL',key,[(round(e,5),l,round(a,4)) for e,l,a,_,_ in fits],flush=True); print(np.round(fits[0][-1].T,3)); F[key]=fits[0][3]
B,C,D=F['v'],F['v1'],F['v2']; X0=np.c_[z,B,C,D];
def score(K,lam):
 out=[]
 for tr,te in splits:
  m=Y[tr].mean(0); E=abs(Y[te]-m-K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),Y[tr]-m,assume_a='pos'));out.append([E[:,0].mean(),E[:,1:].mean()])
 return np.array(out)
R=[]
def add(name,K):
 for lam in [1e-5,.0003,.003,.03,.15]:
  cv=score(K,lam);R.append((name,lam,cv))
for name,u in [('z',z),('B',B),('C',C),('D',D),('BCD',np.c_[B,C,D]),('all',X0),('BC',np.c_[B,C]),('BD',np.c_[B,D])]:
 us=std(u); add(name+' lin',1+us@us.T/u.shape[1]);
 for d in [2,3,4]:add(name+f' poly{d}',(1+us@us.T/u.shape[1])**d)
 ds=cdist(us,us,'sqeuclidean')/u.shape[1]
 for a in [.06,.15,.35,.75]:add(name+f' rbf{a}',np.exp(-ds*a))
 for a in [.5,1,1.5,2,3]:
  f=std(np.c_[u,np.tanh(a*u),np.sin(a*u),u**2,u**3]);add(name+f' additive{a}',1+f@f.T/f.shape[1]);
for order in range(2):
 print('RANK',order)
 for name,l,e in sorted(R,key=lambda r: r[2][:,order].mean())[:12]:print(name,l,np.round(e.T,4), 'mean',np.round(e.mean(0),4))
np.savez('/tmp/readoutcoords.npz',z=z,B=B,C=C,D=D)
