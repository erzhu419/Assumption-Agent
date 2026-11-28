import itertools,pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from panel_ab import fit_a,inv_a
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(p['panel_b'][k] for k in ['u','v'])
t=inv_a(p['train']['x'],fit_a(p['panel_a'])); Y=p['train']['traj']; pairs=list(itertools.combinations_with_replacement(range(4),2)); np.set_printoptions(precision=4,suppress=True)
def basis(x):return np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in pairs]])
def mat(x,f):return np.column_stack((basis(x),np.sin(x*f)))
def fit(x,y,f):
 X=mat(x,f); c=np.linalg.lstsq(X,y,rcond=None)[0]; return X@c-y,c
for lo,hi in [(0.3,2),(2,3.5),(3.5,5),(5,7)]:
 f=minimize_scalar(lambda f:np.linalg.norm(fit(u,v,f)[0]),bounds=(lo,hi),method='bounded').x; e,c=fit(u,v,f); X=mat(u,f); loo=np.abs(e/(1-(np.linalg.qr(X)[0]**2).sum(1))[:,None]); Z=mat(t,f)@c; A=np.c_[np.ones(len(t)),Z]; print('shared f/RMS/max LOO/trajectory MAE',f,np.sqrt(np.mean(e**2)),loo.max(),np.mean(abs(A@np.linalg.lstsq(A,Y,rcond=None)[0]-Y))); print(c)
f=least_squares(lambda f:fit(u,v,f)[0].ravel(),np.full(4,3.),bounds=(.1,7)).x; e,c=fit(u,v,f); print('coordinate freqs',f,'RMS',np.sqrt(np.mean(e**2))); print(c)
for j in range(4):
 print('output',j); models=[]
 for k in range(4):
  for nh in range(3):
   for inds in itertools.combinations(range(10),nh):
    cols=[*range(5),*[5+i for i in inds],15+k]; b=basis(u)[:,cols[:-1]]
    def objective(f):
     a=np.c_[b,np.sin(f*u[:,k])]; return np.linalg.norm(a@np.linalg.lstsq(a,v[:,j],rcond=None)[0]-v[:,j])
    f=minimize_scalar(objective,bounds=(.3,5),method='bounded').x; models.append((objective(f),k,[pairs[i] for i in inds],f))
 print(sorted(models)[:5])
