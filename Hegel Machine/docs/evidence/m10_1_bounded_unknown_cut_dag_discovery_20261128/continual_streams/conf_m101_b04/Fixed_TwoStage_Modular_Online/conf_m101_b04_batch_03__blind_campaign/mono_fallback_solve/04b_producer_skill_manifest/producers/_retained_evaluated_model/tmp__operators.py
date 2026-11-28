import pickle,numpy as np,sys
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task'); import batch_solver as bs
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; a=bs._calibration(s['panel_a']); z=(x-a[0])@np.linalg.inv(a[1:]); t=s['eval_times']; Y=np.c_[tr['y'],tr['traj']];
def fit(f,y):return np.linalg.lstsq(f,y,rcond=None)[0]
def lin(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,c],axis=1) for k in range(d+1) for c in combinations_with_replacement(range(x.shape[1]),k)])
def op(u,v):
 for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:
  def calc(w,X=u):return np.c_[lin(X),f(w*X)]
  def cost(w):
   F=calc(w); E=(v-F@fit(F,v))/(1-np.sum(np.linalg.qr(F)[0]**2,axis=1))[:,None]; return np.mean(E**2)**.5
  q=minimize_scalar(cost,bounds=(.2,3.5),method='bounded'); yield q.fun,name,q.x,lambda x,f=f,w=q.x,b=fit(calc(q.x),v):np.c_[lin(x),f(w*x)]@b
ops={}
for key,o in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[key]; r=sorted(op(p['u'],p[o])); print(key,o,[(round(e,6),n,round(w,5)) for e,n,w,_ in r]); ops[o]=r[0][-1]
Q=np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2].T; yy=np.c_[Y[:,0],Y[:,1:]@Q]; rng=np.random.default_rng(710); ids=[np.arange(72),*[rng.permutation(72) for _ in range(8)],*[np.argsort(abs(z[:,j])) for j in range(4)]]
def test(F,lam=0):
 F=F/np.maximum(np.std(F,axis=0),.05); out=[]
 for k,ix in enumerate(ids):
  i,v=ix[:(56 if 0<k<9 else 52)],ix[(56 if 0<k<9 else 52):]; g=F[i]; b=fit(g,yy[i]) if lam==0 else np.linalg.solve(g.T@g+lam*np.eye(g.shape[1]),g.T@yy[i]); E=F[v]@b-yy[v]; out.append([abs(E[:,0]).mean(),abs(E[:,1:]@Q.T).mean()]);
 return np.round(np.r_[out[0],np.mean(out[1:9],axis=0),np.max(out[9:],axis=0)],4)
V={k:f(z) for k,f in ops.items()}; V2={k:f(x) for k,f in ops.items()}; b=bs._features(z)
for name,X in [('z',z),*V.items(),*((k+'raw',v) for k,v in V2.items()),('v2ofv',ops['v2'](V['v'])),('vofv1',ops['v'](V['v1']))]:
 for m,F in [('quad',poly(X,2)),('b+lin',np.c_[b,X]),('b+quad',np.c_[b,poly(X,2)[:,1:]]),('b+tanh',np.c_[b,np.tanh(X)])]:print(name,m,test(F,.02))
for w in [.55,.7,.85,1,1.15,1.3,1.5,1.8]:
 print('tanh',w,test(np.c_[poly(z,2),np.tanh(w*z)],.002),'sin',test(np.c_[poly(z,2),np.sin(w*z)],.002))
