import importlib.util,pickle,numpy as np,itertools as it
from scipy.linalg import cho_factor,cho_solve
p=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];z,b=m._coordinates(s,tr['x']);N=len(z);R=np.random.default_rng(713);Y=np.c_[tr['y'],tr['traj']]; Y[:,1:]=Y[:,1:]@np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2].T@np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2]
F=[np.arange(52,N)]+[R.permutation(N)[:20] for _ in range(12)]; ps=[];names=[];Ks=[];regs=[]
for name in ['tanh','erf','rat','mix']:
 for sz,sb in it.product([1.,1.5,2.,2.5,3.,4.],repeat=2):
  if name=='tanh':hz,hb=np.tanh(sz*z),np.tanh(sb*b)
  if name=='erf':hz,hb=m.erf(sz*z),m.erf(sb*b)
  if name=='rat':hz,hb=sz*z/np.sqrt(1+(sz*z)**2),sb*b/np.sqrt(1+(sb*b)**2)
  if name=='mix':hz,hb=np.c_[np.tanh(sz*z),m.erf(sz*z),sz*z/np.sqrt(1+(sz*z)**2)],np.c_[np.tanh(sb*b),m.erf(sb*b),sb*b/np.sqrt(1+(sb*b)**2)]
  for raw in [.3,1.]:
   x=np.c_[z,b,hz,hb];x=x/x.std(0);x[:,:8]*=raw;L=x@x.T/x.shape[1]; S=(1+L)**2;C=(1+L)**3
   for wt in [.15,1.]:
    ker=m._kernel(x,x);Q=ker-S
    for cube in [0,.25]:
     K=S+wt*Q+cube*C
     for reg in [.00015,.001,.004]:
      es=[]
      for j in F:
       i=np.setdiff1d(np.arange(N),j,assume_unique=False); pred=K[np.ix_(j,i)]@cho_solve(cho_factor(K[np.ix_(i,i)]+reg*np.eye(len(i))),Y[i]);d=abs(pred-Y[j]);es.append([d[:,0].mean(),d[:,1:].mean(),d[:,-1].mean()]);
      es=np.array(es);ps.append(es);names.append((name,sz,sb,raw,wt,cube,reg))
P=np.array(ps); med=P[:,1:].mean(1);mx=P[:,1:].max(1)
for w in [[1,1,0],[1,1,1],[1,0,0],[0,0,1]]:
 ix=np.argsort((med+.5*mx)@w)
 for k in ix[:5]:print(names[k],np.round(P[k,0],5),np.round(med[k],5),np.round(mx[k],5))
print('base',[(names[k],P[k,0],med[k],mx[k]) for k in range(len(P)) if names[k]==('tanh',2.,2.,1.,1.,0,.001)])
