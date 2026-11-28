import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from panel_ab import fit_a,inv_a
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(p['panel_b'][k] for k in ('u','v')); t=inv_a(p['train']['x'],fit_a(p['panel_a'])); Y=p['train']['traj']; cut=len(t)-20
pairs=list(itertools.combinations_with_replacement(range(4),2)); np.set_printoptions(precision=4,suppress=True)
def q(x): return np.array([x[:,i]*x[:,j] for i,j in pairs]).T
def aff(x):return np.c_[np.ones(len(x)),x]
def cv(z):
 A=aff(z); return np.mean(abs(A[cut:]@np.linalg.lstsq(A[:cut],Y[:cut],rcond=None)[0]-Y[cut:]))
shapes={
 'cubic':lambda x,f:x**3,
 'quintic':lambda x,f:x**5,
 'rational':lambda x,f:x**3/(1+f*x*x),
 'rational5':lambda x,f:x**5/(1+f*x*x),
 'sine':lambda x,f:np.sin(f*x),
 'tanh':lambda x,f:np.tanh(f*x),
 'atan':lambda x,f:np.arctan(f*x),
 'exp':lambda x,f:x*np.exp(-f*x*x),
 'sqrt':lambda x,f:x/np.sqrt(1+f*x*x),
 'signed_sq':lambda x,f:x*abs(x)}
for name,h in shapes.items():
 for mode in ('none','range','quadratic','cyclic'):
  def features(x,f):
   H=h(x,f[0]); Q=q(x) if mode=='quadratic' else (x*np.roll(x,-1,axis=1) if mode=='cyclic' else np.empty((len(x),0)))
   return np.c_[aff(x),H,Q] if mode!='range' else np.c_[np.ones(len(x)),x+f[1]*H]
  def fit(f):
   A=features(u,f); C=np.linalg.lstsq(A,v,rcond=None)[0]; return A@C-v,C
  candidates=[]
  for f0 in ([.3,1.2,2.7,4.] if name not in ('cubic','quintic','signed_sq') else [1.]):
   for a in ([-.5,.5] if mode=='range' else [.5]):
    sol=least_squares(lambda f:fit(f)[0].ravel(),[f0,a] if mode=='range' else [f0],bounds=([.01,-20],[8.,20]) if mode=='range' else ([.01],[8.]),max_nfev=150)
    e,C=fit(sol.x); candidates.append((np.linalg.norm(e),sol.x,C))
  er,f,C=min(candidates,key=lambda w:w[0]); score=cv(features(t,f)@C); print(name,mode,'fit',round(er/np.sqrt(v.size),6),'cv',round(score,6),'f',f)
  if score<.05:print(C)
# Test alternative sequencing without relying on a training-label inverse.
U=inv_a(u,fit_a(p['panel_a'])); V=inv_a(v,fit_a(p['panel_a'])); print('A-inverted B input/output ranges',U.min(0),U.max(0),V.min(0),V.max(0)); print('affine sequence probes',[(tag,float(np.mean(abs(aff(a)@np.linalg.lstsq(aff(a),b,rcond=None)[0]-b)))) for tag,a,b in [('u->v',u,v),('A^-1 u->v',U,v),('u->A^-1 v',u,V),('A^-1 u->A^-1 v',U,V)]])
