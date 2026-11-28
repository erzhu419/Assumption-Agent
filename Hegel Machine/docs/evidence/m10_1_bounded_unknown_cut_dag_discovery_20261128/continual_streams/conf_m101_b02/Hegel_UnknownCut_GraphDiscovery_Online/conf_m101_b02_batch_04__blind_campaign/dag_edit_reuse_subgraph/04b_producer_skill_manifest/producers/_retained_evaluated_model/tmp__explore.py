import sys,pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from sklearn.model_selection import KFold,cross_val_predict
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=node_00.run_node({'__input__':s}); z,b=f['z'],f['b']; u=s['panel_c']['u']; v=s['panel_c']['v2']; t=s['eval_times']; y=s['train']['y']; traj=s['train']['traj']; n=len(y)
def design(*xs):
 return np.column_stack((np.ones(len(xs[0])),*xs))
def quad(x): return np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def loo(x,y):
 e=y-x@fit(x,y);return np.mean((e/(1-np.sum(x*np.linalg.pinv(x).T,axis=1))[:,None])**2)**.5
funcs={'sq':lambda x,k:x*x,'cos':lambda x,k:np.cos(k*x),'exp':lambda x,k:np.exp(-k*x*x),'rat':lambda x,k:1/(1+k*x*x),'sqrt':lambda x,k:np.sqrt(1+k*x*x),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'u*exp':lambda x,k:x*np.exp(-k*x*x)}
for name,fn in funcs.items():
 opt=minimize_scalar(lambda k:loo(design(u,fn(u,k)),v),bounds=(.01,5),method='bounded');print('C2',name,opt.x,opt.fun)
 opt=minimize_scalar(lambda k:loo(design(fn(u,k)),v),bounds=(.01,5),method='bounded');print('  warp-only',opt.x,opt.fun)
c=design(quad(z))@fit(design(quad(u)),s['panel_c']['v1']);d=design(z,z*z)@fit(design(u,u*u),v)
# Profile an exponential approach to equilibrium from zero.
def law(k):return -np.expm1(-k*t)/k
dr=[]
for row in traj:
 def profile(k):
  p=law(k);a=row@p/(p@p);return np.mean((row-a*p)**2),a
 opt=minimize_scalar(lambda k:profile(k)[0],bounds=(.01,2),method='bounded');m,a=profile(opt.x);dr.append((a,opt.x,m**.5))
dr=np.array(dr); print('trajectory profile quantiles',np.quantile(dr,[0,.1,.5,.9,1],axis=0));print('rate corr',np.round(np.corrcoef(np.c_[y,dr,z[:n],b[:n],c[:n],d[:n]].T)[2],3))
folds=list(KFold(5,shuffle=True,random_state=13).split(y))+[(np.arange(n-20),np.arange(n-20,n))]
def cv(X,Y,inv=lambda x:x):
 return np.array([np.mean(abs(inv(X[j]@fit(X[i],Y[i]))-inv(Y[j])),axis=0) for i,j in folds])
for label,xx in [('z',[z]),('b',[b]),('c',[c]),('d',[d]),('bc',[b,c]),('cd',[c,d]),('all',[z,b,c,d]),('zquad',[z,quad(z)]),('zbc',[z,b,c]),('nonlinear',[z,b,c,d,np.tanh(b),np.tanh(c),np.tanh(d)])]:
 X=design(*xx)[:n]; a=cv(X,np.c_[y,dr[:,:2],traj[:,-1]]);print(label,'med',np.round(np.median(a,axis=0),4),'tail',np.round(a[-1],4),'max',np.round(a.max(axis=0),4))
for name, target in [('y',y),('drive',dr[:,0]),('rate',dr[:,1])]:
 for cat,arr in [('z',z),('b',b),('c',c),('d',d)]:
  print(name,cat,np.round(np.corrcoef(np.c_[target,arr[:n]].T)[0,1:],3))
