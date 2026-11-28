import importlib.util,pickle,numpy as np
from itertools import combinations
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import least_squares

def load(i):
 s=importlib.util.spec_from_file_location('node%d'%i,'/holdout_task/node_%02d.py'%i);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x,y,z=[s['train'][k] for k in ('x','y','traj')];Y=np.c_[y,z]; f=load(0).run_node({'__input__':s})['cubic_train']; l=np.cbrt(f);n=len(f); rng=np.random.default_rng(3411); splits=[(np.arange(52),np.arange(52,n))]; splits.extend((p[20:],p[:20]) for p in [rng.permutation(n) for _ in range(12)]);splits.extend((np.argsort(v)[:-14],np.argsort(v)[-14:]) for v in abs(l).T)

def test(K,pen,train,test):
 mu=Y[train].mean(0); A=K[np.ix_(train,train)]+pen*np.eye(len(train));return mu+K[np.ix_(test,train)]@cho_solve(cho_factor(A),Y[train]-mu)

def score(pred):
 E=[abs(pred(i,j)-Y[j]) for i,j in splits];M=np.array([[v[:,0].mean(),v[:,1:].mean()] for v in E]);return np.r_[M[0],M[1:13].max(0),M[13:].max(0),np.max([v[:,1:].mean(0) for v in E[1:13]])]

m=load(1); base=lambda i,j:m.run_node({'__input__':{'train':{'y':y[i],'traj':z[i]}},'node_00':{'cubic_train':f[i],'cubic_test':f[j]}});print('Exact current candidate public, random maxima, tail maxima, worst horizon:',score(lambda i,j:np.column_stack(tuple(base(i,j)[k] for k in ('y_hat','state_pred')))))
for name,w in [('cub',f),('cube+lat',np.c_[f,l]),('cube+lat3',np.c_[f,.3*l]),('cube+quint',np.c_[f,.5*l**5])]:
 w=w/(np.std(f,axis=0).mean()); sq=(w[:,None]-w[None,:])**2;g=w@w.T/w.shape[1];a=[]
 for d in [1,2,3,4,5]:
  for mix in [0,.5,.9]:
   K=(1-mix)*(1+g)**d+mix*(1+w[:,None,:]*w[None,:,:]) .__pow__(d).mean(2)
   for p in [1e-6,1e-4,.001,.01,.05]:a.append((score(lambda i,j:test(K,p,i,j)),('poly',d,mix,p)))
 for gamma in [.002,.007,.015,.03,.06,.12,.25]:
  for mix in [.2,.5,.8,.95]:
   K=(1-mix)*np.exp(-gamma*sq.mean(2))+mix*np.exp(-gamma*sq).mean(2)
   for p in [1e-6,1e-5,3e-5,1e-4,3e-4,.001]:a.append((score(lambda i,j:test(K,p,i,j)),('rbf',gamma,mix,p)))
 print(name,'top random',[(v.round(4).tolist(),h) for v,h in sorted(a,key=lambda q:max(q[0][:4]))[:3]],'top combined',[(v.round(4).tolist(),h) for v,h in sorted(a,key=lambda q:max(q[0][:6]))[:3]])
print('cubic affine betas',np.linalg.lstsq(np.c_[np.ones(n),f],Y,rcond=None)[0].round(3))
