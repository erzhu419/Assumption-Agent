import itertools,pickle,numpy as np
from scipy.optimize import least_squares
from panel_ab import fit_a,inv_a
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(p['panel_b'][k] for k in ('u','v')); t=inv_a(p['train']['x'],fit_a(p['panel_a'])); Y=p['train']['traj']; cut=len(t)-20
np.set_printoptions(precision=4,suppress=True)
def aff(x):return np.c_[np.ones(len(x)),x]
def score(z):
 A=aff(z); c=np.linalg.lstsq(A[:cut],Y[:cut],rcond=None)[0]; return np.mean(abs(A[cut:]@c-Y[cut:])),np.linalg.norm(c[1:])
pairs=list(itertools.combinations_with_replacement(range(4),2)); cyc=[(i,(i+1)%4) for i in range(4)]; opp=[(0,2),(1,3)]; sq=[(i,i) for i in range(4)]
for shape in ['sin','rational','tanh']:
 def h(x,a,f):
  if shape=='sin':return x+a*np.sin(f*x)
  if shape=='rational':return x+a*x/(1+f*x*x)
  return x+a*np.tanh(f*x)
 for mode in ['raw','warped','mixed','squared_sum']:
  for label,pr in [('cyc',cyc),('opp',opp),('sq',sq),('all',pairs)]:
   def basis(x,par):
    a,f=par; w=h(x,a,f)
    if mode=='raw':Q=[x[:,i]*x[:,j] for i,j in pr]
    if mode=='warped':Q=[w[:,i]*w[:,j] for i,j in pr]
    if mode=='mixed':Q=[x[:,i]*w[:,j]+w[:,i]*x[:,j] for i,j in pr]
    if mode=='squared_sum':Q=[h(x[:,i]+x[:,j],a,f)-(x[:,i]+x[:,j]) for i,j in pr]
    return np.c_[aff(w),*Q]
   def fit(par):
    A=basis(u,par); c=np.linalg.lstsq(A,v,rcond=None)[0]; return A@c-v,c
   ff={'sin':2.8,'rational':.8,'tanh':1.4}[shape]; aa={'sin':-.4,'rational':-.9,'tanh':-.8}[shape]; res=least_squares(lambda q:fit(q)[0].ravel(),[aa,ff],bounds=([-4,.1],[2,7]),max_nfev=200); e,c=fit(res.x); print(shape,mode,label,res.x,np.sqrt(np.mean(e**2)),score(basis(t,res.x)@c))
   if np.sqrt(np.mean(e**2))<.001:print(c)
# Check whether the remaining discrepancy is localized or spread across the coordinate range.
w=lambda x:x-.4*np.sin(2.8*x)
Z=aff(w(t)); C=np.linalg.lstsq(Z,Y,rcond=None)[0]; e=(Y-Z@C)[:,-1]; print('trajectory split size and coeff',len(t),cut,C[:,-1]); print('largest residuals, latent and measured'); ids=np.argsort(abs(e))[-12:]; print(np.c_[ids,e[ids],t[ids],p['train']['x'][ids]])
print('A and B calibration sizes',len(p['panel_a']['u']),len(u))
