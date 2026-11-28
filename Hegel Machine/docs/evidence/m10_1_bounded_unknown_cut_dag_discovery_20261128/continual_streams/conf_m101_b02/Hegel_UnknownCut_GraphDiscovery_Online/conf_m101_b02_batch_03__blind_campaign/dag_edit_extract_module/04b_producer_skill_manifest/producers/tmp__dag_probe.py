import sys,pickle,itertools as it,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task')
import node_02
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; n=len(x); z=node_02.run_node({'__input__':s})['z'][:n]; t=s['eval_times']; T=s['train']['traj']; y=s['train']['y']; L=lambda x:np.c_[np.ones(len(x)),x]
p=s['panel_b']; B=np.linalg.lstsq(L(p['u']),p['v'],rcond=None)[0]; b=L(z)@B; f=L(x)@B
p=s['panel_c']; u=p['u']; cs=[]
for key,fun,lim in [('v1',lambda x,k:x/(1+k*abs(x)),(.1,2)),('v2',lambda x,k:x*x/(1+k*x*x),(.02,1))]:
 def obj(k,outs=False):
  X=np.c_[L(u),fun(u,k)]; c=np.linalg.lstsq(X,p[key],rcond=None)[0]; return c if outs else np.mean((X@c-p[key])**2)
 k=minimize_scalar(obj,bounds=lim,method='bounded',options={'xatol':1e-12}).x; cs.append(np.c_[L(z),fun(z,k)]@obj(k,True)); print('C',key,k,np.sqrt(obj(k)),flush=True)
c,d=cs
U,S,V=np.linalg.svd(T,full_matrices=False); print('SVD',S,'Y range',y.min(),y.max(),'B',B,flush=True); print('time',t);print('T rows',T[:5]); print('T right vectors',V[:4]); print('corr',np.corrcoef(np.c_[y,T@V[:3].T].T));
if '--setup' in sys.argv: quit()
fun={'log':lambda t,k:np.log1p(k*t),'exp':lambda t,k:-np.expm1(-k*t),'rat':lambda t,k:t/(1+k*t),'root':lambda t,k:np.sqrt(1+k*t)-1,'sin':lambda t,k:np.sin(k*t),'cos':lambda t,k:1-np.cos(k*t),'pow':lambda t,k:t**k,'lin':lambda t,k:t,'quad':lambda t,k:t*t,'atan':lambda t,k:np.arctan(k*t),'tanh':lambda t,k:np.tanh(k*t),'one':lambda t,k:np.ones_like(t)}
for nt in [2,3]:
 res=[]
 for keys in it.combinations_with_replacement(fun,nt):
  def r(k):
   X=np.column_stack([fun[a](t,v) for a,v in zip(keys,k)]); C=np.linalg.lstsq(X,T.T,rcond=1e-9)[0]; return (T.T-X@C).ravel()
  q=least_squares(r,np.linspace(.7,1.4,nt),bounds=(.02,6.),max_nfev=36);res.append((np.sqrt(np.mean(q.fun**2)),keys,q.x))
 print('time best',nt,sorted(res)[:8],flush=True)
Y=np.c_[y,T]; rng=np.random.default_rng(41); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(rng.permutation(n),4)]
def score(P):
 return np.round([np.mean(np.abs(P[:,0])),np.mean(np.abs(P[:,1:]))],5)
def metrics(predict):
 E=[score(predict(a,i)-Y[i]) for a,i in folds]; return E[0],np.max(E[1:],axis=0)
rows=[]
for name,q in [('z',z),('raw',x),('zB',np.c_[z,b]),('zC',np.c_[z,c,d]),('zBC',np.c_[z,b,c,d]),('zBCrawB',np.c_[z,b,c,d,f]),('B',b),('C',np.c_[c,d])]:
 q=(q-q.mean(0))/q.std(0); K=cdist(q,q,'sqeuclidean')/q.shape[1]; P=q@q.T/q.shape[1]
 for mode in ['rbf','matern','p3','p4','mix']:
  for ls in [.45,.85,1.4,2.4,4.]:
   D=np.sqrt(K)*ls
   M={'rbf':np.exp(-D*D/2),'matern':(1+D+D*D/3)*np.exp(-D),'p3':(1+P/ls)**3,'p4':(1+P/ls)**4,'mix':(1+P/ls)**3+np.exp(-D*D/2)}[mode]
   for lam in [1e-6,.0001,.003,.03,.2]:
    def pr(a,i):
     m=Y[a].mean(0);return m+M[np.ix_(i,a)]@np.linalg.solve(M[np.ix_(a,a)]+lam*np.eye(len(a)),Y[a]-m)
    A,W=metrics(pr);rows.append((max(A),name,mode,ls,lam,A,W))
for k in range(2): print('kernel best target',k,sorted(rows,key=lambda r:r[-2][k])[:5]); print('kernel best worst fold',k,sorted(rows,key=lambda r:r[-1][k])[:4]);
