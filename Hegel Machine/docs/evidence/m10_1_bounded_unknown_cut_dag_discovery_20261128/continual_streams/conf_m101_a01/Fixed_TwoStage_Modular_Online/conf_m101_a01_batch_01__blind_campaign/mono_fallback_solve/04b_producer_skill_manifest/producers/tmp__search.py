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
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from sklearn.linear_model import lars_path
rng=np.random.default_rng(39); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(7)]]
coords={'x':x,'A':ai(x),'a':a(x),'b':b(x),'bA':b(ai(x)),'ba':b(a(x)),'B':bi(x),'BA':bi(ai(x))}
np.set_printoptions(precision=4,suppress=True)
print('quad coefficients',np.linalg.lstsq(quad(x),T,rcond=None)[0].T)
print('linear residuals: x and y,z'); lr=aff(x)@np.linalg.lstsq(aff(x),T,rcond=None)[0]; print(np.c_[x,T-lr][np.argsort(np.abs(y-lr[:,0]))[-12:]])
def test(K,D,lam):
 out=[]
 for i,j in splits:
  H=solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),np.c_[T[i],D[i]],assume_a='pos'); F=H[:,2:]; C=np.linalg.lstsq(D[i].T@F,D[i].T@H[:,:2],rcond=1e-10)[0]; pred=D[j]@C+K[np.ix_(j,i)]@(H[:,:2]-F@C); out.append(np.mean(abs(pred-T[j]),axis=0))
 return np.array(out)
out=[]
for k,u in coords.items():
 for mode in ('isotropic','component','pairs'):
  v=u if mode!='pairs' else np.c_[u,*[(u[:,i]+sg*u[:,j])/2**.5 for i,j in itertools.combinations(range(4),2) for sg in (-1,1)]]
  ds=abs(v[:,None,:]-v[None,:,:]) if mode!='isotropic' else cdist(v,v)[:,:,None];
  for kf in ('m32','m52','sq','rq'):
   for scale in (.35,.65,1.1,1.8,3.0):
    d=ds/scale; K={'m32':lambda:(1+d)*np.exp(-d),'m52':lambda:(1+d+d*d/3)*np.exp(-d),'sq':lambda:np.exp(-d*d/2),'rq':lambda:1/(1+d*d/2)}[kf]().mean(2)
    for trend,D in [('aff',aff(x)),('quad',quad(x)),('full3',np.c_[quad(x),*[np.prod(x[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]] )]:
     for lam in (.0002,.003,.03):
      e=test(K,D,lam);out.append((e.mean(0),e.max(0),e[0],(k,mode,kf,scale,trend,lam)))
for l in range(2):print('kernel',l,*sorted(out,key=lambda q:q[0][l])[:9],sep='\n')
# Projected sparse dictionary: sum/difference, multiplicative, and rational terms.
F=[]; labels=[]
for k,u in coords.items():
 for nm,v in [(str(i),u[:,i]) for i in range(4)]+[(f'{i}{sg:+}*{j}',(u[:,i]+sg*u[:,j])/2**.5) for i,j in itertools.combinations(range(4),2) for sg in (-1,1)]:
  for op,w in [('abs',abs(v)), ('cube',v**3),('soft',v/np.sqrt(1+v*v)),('evenrat',1/(1+v*v)),('sin',np.sin(1.6*v)),('cos',np.cos(1.6*v)),('tanh',np.tanh(1.6*v))]:
   F.append(w); labels.append(f'{k} {nm} {op}')
 for i,j in itertools.permutations(range(4),2):
  for op,w in [('rat',u[:,i]/(1+u[:,j]**2)),('sat',u[:,i]*np.tanh(u[:,j]))]:F.append(w);labels.append(f'{k} {i} {j} {op}')
F=np.array(F).T
for l in range(2):
 for name,D in [('aff',aff(x)),('quad',quad(x))]:
  es=[]; sel=[]
  for i,j in splits:
   coef=np.linalg.lstsq(D[i],np.c_[T[i,l],F[i]],rcond=None)[0]; G=np.c_[T[:,l],F]-D@coef; scale=np.linalg.norm(G[i,1:],axis=0); G[:,1:]/=scale; al,act,path=lars_path(G[i,1:],G[i,0],method='lasso',max_iter=35); es.append([np.mean(abs(G[j,1:]@c-G[j,0])) for c in path.T]);sel.append([labels[v] for v in act[:9]])
  E=np.array([v[:min(map(len,es))] for v in es]); p=np.argmin(E.mean(0));print('sparse',l,name,p,E.mean(0)[p],E.max(0)[p],E[0,p],sel[:3])
