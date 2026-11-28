import pickle,itertools
import numpy as np
from scipy.optimize import minimize_scalar
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def terms(u):
 return np.array([u[:,i]*u[:,j] for i,j in [(0,1),(0,3),(1,2),(2,3)]]).T
def scores(X,y):
 P=np.linalg.pinv(X); c=P@y; e=y-X@c; h=np.sum(X*P.T,axis=1); return np.mean(abs(e/(1-h[:,None]))),c,np.mean(abs(e))
func={'sin':np.sin,'tanh':np.tanh,'quad':lambda x:x*x,'cos':lambda x:1-np.cos(x),'rat':lambda x:x/(1+x*x),'cubic':lambda x:x**3}
np.set_printoptions(precision=4,suppress=True,linewidth=150)
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v2')]:
 u=S[p]['u']; y=S[p][k]; base=np.c_[np.ones(len(u)),u]
 if p=='panel_a':
  w=np.ones(len(u));
  for i in range(15):
   c=np.linalg.lstsq(base*w[:,None],y*w[:,None],rcond=None)[0]; r=np.linalg.norm(y-base@c,axis=1); w=1/np.sqrt(1+(r/.15)**2)
  good=np.argsort(r)[:-3]; u,y,base=u[good],y[good],base[good]
 print('\n',p,'aff',scores(base,y)[0]); opts=[]
 for (a,b) in itertools.combinations_with_replacement(func,2):
  def mat(t):
   aa=func[a](t*u); bb=func[b](t*u); return np.c_[base,aa] if a==b else np.c_[base,aa,bb]
  fit=minimize_scalar(lambda t:scores(mat(t),y)[0],bounds=(.51,2.5),method='bounded'); opts.append((fit.fun,a,b,fit.x,scores(mat(fit.x),y)[2]))
 print(*sorted(opts)[:10],sep='\n')
 for desc,X in [('quadratic',np.c_[base,u*u,terms(u),u[:,0]*u[:,2],u[:,1]*u[:,3]]),('sin/cos 1',np.c_[base,np.sin(u),1-np.cos(u)]),('tanh 1',np.c_[base,np.tanh(u),np.tanh(u)**2]),('diag cubic',np.c_[base,u**2,u**3])]:
  e,c,f=scores(X,y); print(desc,e,f); 
  if p=='panel_a' and desc in ('quadratic','diag cubic'):print(c)
