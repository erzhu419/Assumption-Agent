import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
from batch_solver import _coordinates,_fit,_affine,_products,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; X=np.r_[tr['x'],s['test']['x']]; Z=_coordinates(s['panel_a'],X)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
F={'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'atan':np.arctan,'rootrat':lambda x:x/np.sqrt(1+x*x),'log':lambda x:np.log1p(x*x)}
def panel(p,k,f,k0,mode):
 u=s[p]['u']; v=s[p][k]; fun=F[f];
 def d(u,b):
  h=fun(u*np.exp(b[:4])+(b[4:] if len(b)>4 else 0)); return np.c_[_affine(u),h]
 def res(b):
  D=d(u,b); E=v-D@_fit(D,v); return E.ravel()
 b=least_squares(res,np.r_[np.repeat(np.log(k0),4),np.zeros(4) if mode else []],bounds=(-3,3),max_nfev=200).x
 D=d(u,b); w=_fit(D,v); e=v-D@w; loo=e/(1-(D*np.linalg.pinv(D).T).sum(1))[:,None]
 print('panel',k,f,mode,'scale',np.exp(b[:4]),'off',b[4:],'rmse/loo',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(loo**2)))
 if not mode: print('w',w.T)
 return d(Z,b)@w
if __name__=='__main__':
 print('traj spectrum',np.linalg.svd(Y[:,1:]-Y[:,1:].mean(0),compute_uv=False),'means',Y[:,1:].mean(0)); print('traj modes',np.linalg.svd(Y[:,1:]-Y[:,1:].mean(0),full_matrices=False)[2][:3]);
 rng=np.random.default_rng(795); perm=rng.permutation(72); folds=[np.arange(52,72),np.arange(56,72),*np.array_split(perm,4),np.argsort(abs(Z[:72]).max(1))[-16:]]
 def metrics(e):return np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
 def test(a):
  err=[]
  for te in folds:
   i=np.setdiff1d(np.arange(72),te); err.append(metrics(abs(a[te]@_fit(a[i],Y[i])-Y[te])))
  return np.round(np.stack([err[0],err[1],np.max(err[2:-1],axis=0),err[-1]]),5)
 def report(name,a):print(name,a.shape[1],test(a).ravel())
 base=np.c_[_affine(Z),_products(Z)]; report('baseline',np.c_[base,np.tanh(Z)]);
 for te in folds:
  i=np.setdiff1d(np.arange(72),te); sub=dict(s,train=dict(tr,x=X[i],y=Y[i,0],traj=Y[i,1:]),test={'x':X[te]}); out=solve_batch(sub); print('artifact',len(te),metrics(abs(np.c_[out['y_hat'],out['state_pred']]-Y[te])))
 for f,k0 in [('tanh',1.4),('rat',.87),('atan',1.3),('rootrat',1.1)]:
  for mode in (0,1):
   b=panel('panel_b','v',f,k0,mode); report('B mix '+f,np.c_[base,b]);report('B products '+f,np.c_[_affine(Z),b,_products(b)])
 for f,k0 in [('atan',2.38),('rootrat',1.89),('tanh',2.2)]:
  for mode in (0,1):
   c=panel('panel_c','v1',f,k0,mode);report('C1 mix '+f,np.c_[base,c]);report('C1 products '+f,np.c_[_affine(Z),c,_products(c)])
 c2=panel('panel_c','v2','log',.602,0);report('baseline + C2',np.c_[base,np.tanh(Z),c2]);
 for f in F:
  if f=='log':continue
  for k in (.7,1.,1.4,1.8,2.2,2.6):report('profile '+f+' '+str(k),np.c_[base,F[f](k*Z)])
