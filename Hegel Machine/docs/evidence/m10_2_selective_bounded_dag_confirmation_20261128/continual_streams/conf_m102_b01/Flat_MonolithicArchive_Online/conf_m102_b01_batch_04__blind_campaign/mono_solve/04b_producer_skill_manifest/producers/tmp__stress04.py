import importlib.util, pickle,itertools
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y'];T=tr['traj'];t=s['eval_times'];z=m._latent(s,np.r_[tr['x'],s['test']['x']]);D=m._quadratic(z);A=m._aff(z)
print('scalar full coeff',m._ls(D[:72],y)); print('scalar first52 coeff',m._ls(D[:52],y[:52])); print('trajectory affine coeff\n',m._ls(A[:72],T)); print('test leverage maxima',*[np.max(np.sum((a[72:]@np.linalg.pinv(a[:72]))**2,axis=1)) for a in (D,A)])
rng=np.random.default_rng(677); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(30)]
# Compare unconstrained quadratic against fixed two-interaction and training-only sparse supports.
def sparse(i,k):
 best=(np.inf,None)
 for ix in itertools.combinations(range(5,15),k):
  inds=list(range(5))+list(ix);d=D[i][:,inds];pin=np.linalg.pinv(d);e=y[i]-d@(pin@y[i]); h=np.sum(d*pin.T,axis=1);loss=np.mean((e/(1-h))**2)
  if loss<best[0]:best=(loss,inds)
 return best[1]
for kind in ['full','pair','affine','sparse1','sparse2','sparse3']:
 scores=[];sets=[]
 for split in splits:
  i,j=split[:52],split[52:]
  ix={'full':list(range(15)),'pair':[0,1,2,3,4,7,11],'affine':list(range(5))}.get(kind)
  if ix is None: ix=sparse(i,int(kind[-1]))
  sets.append(tuple(ix));d=D[:,ix];e=d[j]@m._ls(d[i],y[i])-y[j];scores.append([np.mean(abs(e)),np.max(abs(e))])
 from collections import Counter
 print(kind,'primary',scores[0],'mean',np.mean(scores,0),'worst',np.max(scores,0),'supports',Counter(sets).most_common(4))
for kind in ['affine','rank2','explog','exp2','log2']:
 scores=[]
 for split in splits:
  i,j=split[:52],split[52:];coef=m._ls(A[i],T[i]);p=A[j]@coef
  if kind=='rank2':
   v=np.linalg.svd(A[i]@coef,full_matrices=False)[2][:2];p=(p@v.T)@v
  elif kind!='affine':
   def basis(q):
    r,k=np.exp(q)
    return np.column_stack((np.log1p(r*t) if kind=='log2' else -np.expm1(-r*t), -np.expm1(-k*t) if kind=='exp2' else np.log1p(k*t)))
   def err(q):
    f=basis(q);return (coef-m._ls(f,coef.T).T@f.T).ravel()
   q=least_squares(err,[.6,0.] if kind!='exp2' else [.6,-1.],bounds=([-5,-5],[4,4])).x; f=basis(q);p=m._ls(f,p.T).T@f.T
  e=np.abs(p-T[j]); scores.append([e.mean(),e.mean(0).max(),e[:,-1].mean(),e.max()])
 print(kind,'primary',scores[0],'mean',np.mean(scores,0),'worst',np.max(scores,0),('primary spatial fit RMSE',np.sqrt(np.mean(err(q)**2)),np.exp(q)) if kind not in ('affine','rank2') else '')
