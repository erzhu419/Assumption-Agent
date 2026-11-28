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
np.set_printoptions(precision=4,suppress=True); print('A B',A,B); print('rate per row',np.quantile([minimize_scalar(lambda r: np.mean((w-(w@f)*f/(f@f))**2) if (f:=-np.expm1(-r*t)) is not None else 0,bounds=(.3,1.4),method='bounded').x for w in tr[np.abs(z)>.3]],[0,.1,.5,.9,1])); print('time residual via SVD',np.linalg.svd(tr-z[:,None]*(-np.expm1(-r*t)),full_matrices=False)[2][:2]);
coords={'x':x,'ax':a(x),'aix':ai(x),'bx':b(x),'bax':b(a(x)),'baix':b(ai(x)),'abx':a(b(x)),'aibx':ai(b(x)),'bix':bi(x),'biax':bi(a(x)),'biaix':bi(ai(x))}
ops={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'cos':np.cos,'rat':lambda u:u/(1+u*u),'abs':np.abs}
report=[]
for k,u in coords.items():
 for label,base in [('linear',aff(x)),('quad',quad(x))]:
  print(k,label,'baseline',evaluate(base)) if k=='x' else None
  for op,f in ops.items():
   for w in [.5,.8,1,1.2,1.6,2,2.5,3]:
    F=f(u*w); E=evaluate(np.c_[base,F]); report.append((E.tolist(),k,label,op,w))
for i in range(2):
 print('target',i); print(*sorted(report,key=lambda q:q[0][i])[:18],sep='\n')
