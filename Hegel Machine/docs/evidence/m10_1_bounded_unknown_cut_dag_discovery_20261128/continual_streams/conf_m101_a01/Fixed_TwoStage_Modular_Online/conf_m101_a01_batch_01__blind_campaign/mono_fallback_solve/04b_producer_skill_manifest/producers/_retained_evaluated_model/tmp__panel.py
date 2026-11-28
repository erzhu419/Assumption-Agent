import pickle,itertools
import numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; n=len(x)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(u,v):return np.linalg.lstsq(aff(u),v,rcond=None)[0]
p=s['panel_a']; g=np.linalg.norm(p['u']-p['v'],axis=1)<1; A=fit(p['u'][g],p['v'][g]); p=s['panel_b']; B=fit(p['u'],p['v']);
def a(z):return aff(z)@A
def ai(z):return (z-A[0])@np.linalg.inv(A[1:])
def b(z):return aff(z)@B
def bi(z):return (z-B[0])@np.linalg.inv(B[1:])
def amp(r):
 f=-np.expm1(-r*t); z=tr@f/(f@f); return z,np.mean((tr-z[:,None]*f)**2)
r=minimize_scalar(lambda r:amp(r)[1],bounds=(.3,1.4),method='bounded').x; z=amp(r)[0]; T=np.c_[y,z];
def quad(x):return np.c_[aff(x),*[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)]]
def evaluate(D,Y=T):
 c=np.linalg.lstsq(D[:52],Y[:52],rcond=None)[0]; return np.mean(abs(D[52:]@c-Y[52:]),axis=0)
from itertools import combinations_with_replacement as cwr
from scipy.optimize import minimize_scalar

def loo(D,Y):
 u,d,v=np.linalg.svd(D,full_matrices=False); u=u[:,d>d[0]*1e-9]; return np.mean(((Y-u@(u.T@Y))/np.maximum(1-np.sum(u*u,axis=1),1e-8)[:,None])**2)**.5

def diagpoly(u,d):return np.c_[aff(u),*[u**k for k in range(2,d+1)]]

def pairs(u):return np.c_[*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]]

p=s['panel_c']; u=p['u']; Y=np.c_[p['v1'],p['v2']]; coeff=np.linalg.lstsq(quad(u),Y,rcond=None)[0]; np.set_printoptions(precision=3,suppress=True); print('C quadratic coefficients',coeff.T)
for o in [p['v1'],p['v2']]:
 print('polynomial LOO:',[(d,loo(diagpoly(u,d),o)) for d in (1,2,3,4)],'affine+cross',loo(np.c_[aff(u),pairs(u)],o))
 out=[]
 for k,h in [('x',lambda u:u),('a',a),('ai',ai),('b',b),('bi',bi),('ba',lambda u:b(a(u))),('bai',lambda u:b(ai(u)))]:
  hu=h(u)
  for j,op in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('softsign',lambda z:z/np.sqrt(1+z*z)),('rat',lambda z:z/(1+z*z)),('square',np.square),('cube',lambda z:z**3),('exp',np.exp),('tanh2',lambda z:np.tanh(z)**2),('abs',np.abs),('elu',lambda z:np.expm1(-z*z))]:
   for baseid,base in [('aff',aff(u)),('diag2',diagpoly(u,2)),('cross',np.c_[aff(u),pairs(u)]),('quad',quad(u))]:
    def score(w):return loo(np.c_[base,op(w*hu)],o)
    res=minimize_scalar(score,bounds=(.15,4),method='bounded',options={'xatol':.002}); out.append((res.fun,k,j,baseid,round(res.x,3)))
 print('best',sorted(out)[:15])
for k in range(1,5):
 D=np.c_[aff(x),*[np.prod(x[:,j],axis=1) for d in range(2,k+1) for j in cwr(range(4),d)]]
 print('raw full polynomial',k,evaluate(D), 'ridge', [np.mean(abs(D[52:]@np.linalg.solve(D[:52].T@D[:52]+lam*np.eye(D.shape[1]),D[:52].T@T[:52])-T[52:]),axis=0) for lam in [.01,.1,1]])
