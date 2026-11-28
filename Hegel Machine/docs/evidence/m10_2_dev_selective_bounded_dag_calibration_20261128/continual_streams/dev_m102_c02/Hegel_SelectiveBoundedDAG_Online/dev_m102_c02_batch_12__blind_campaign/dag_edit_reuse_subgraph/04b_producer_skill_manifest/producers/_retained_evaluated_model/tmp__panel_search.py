import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_b'][k] for k in ('u','v')); one=np.ones(len(u)); i,j=np.triu_indices(4,1)
fams={'cos':lambda x,a:np.cos(a*x),'exp':lambda x,a:np.exp(-a*x*x),'rational':lambda x,a:1/(1+a*x*x),'tanh2':lambda x,a:np.tanh(a*x)**2,'cos_all':lambda x,a:np.c_[np.cos(a*x),np.sin(a*x)],'sin':lambda x,a:np.sin(a*x),'abs_power':lambda x,a:abs(x)**a,'cosquad':lambda x,a:np.cos(a*x*x),'sinc':lambda x,a:np.sinc(a*x/np.pi),'sech':lambda x,a:1/np.cosh(a*x),'arctan2':lambda x,a:np.arctan(a*x)**2}
def metric(A):
 r=v-A@np.linalg.lstsq(A,v,rcond=None)[0]; h=np.sum(np.linalg.qr(A,mode='reduced')[0]**2,axis=1); return np.abs(r/np.maximum(1-h[:,None],1e-12)).mean(), np.sqrt(np.mean(r*r))
for name,f in fams.items():
 for mode in range(2):
  def X(a): return np.c_[one,f(u,a)] if not mode else np.c_[one,u,f(u,a)]
  grid=np.linspace(.1,12,160); ix=np.argmin([metric(X(a))[0] for a in grid]); a=minimize_scalar(lambda a:metric(X(a))[0],bounds=(grid[max(0,ix-1)],grid[min(ix+1,len(grid)-1)]),method='bounded').x
  print(name,mode,round(a,5),metric(X(a)),'at 2.15',metric(X(2.15)))
print('poly even:',[(d,metric(np.column_stack([one]+[u**k for k in range(2,d+1,2)]))) for d in [2,4,6,8]])
for a in [2.15,4.3,np.pi,2.15*np.pi]:
 X=np.c_[one,u,np.cos(a*u)]; C=np.linalg.lstsq(X,v,rcond=None)[0]; print('cos',a,'coeff',C.round(4))
