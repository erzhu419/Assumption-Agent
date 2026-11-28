import importlib.util,numpy as np,pickle
from scipy.optimize import least_squares
q=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(q);q.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; raw=m._features(s,tr['x']); dot=raw[:,[9,14,19,24]].sum(1); off=raw[:,10]; F=np.c_[raw[:,:9],dot,off]; T=tr['traj']; np.set_printoptions(precision=5,suppress=True)
models={'both':(list(range(11)),list(range(11))),'disjoint':([1,2,3,4,9,10],[5,6,7,8,10]),'disjoint+bias':([0,1,2,3,4,9,10],[0,5,6,7,8,10]),'shared+bias':(list(range(11)),[0,5,6,7,8,10]),'no-off':([0,1,2,3,4,9],[0,5,6,7,8,10])}
def fit(F,T,t,fast,slow):
 def mat(r):
  h=-np.expm1(-t[:,None]*r)
  return np.column_stack([np.einsum('ij,k->ikj',F[:,ids],h[:,j]).reshape((len(F)*len(t),len(ids))) for j,ids in enumerate((fast,slow))])
 def res(r):
  D=mat(r);return D@m.ls(D,T.ravel())-T.ravel()
 r=least_squares(res,[1.83,.41],bounds=([.85,.03],[5.,.84]),ftol=1e-11).x
 return r,m.ls(mat(r),T.ravel()),np.mean(abs(res(r)))
partitions=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(10)]+[np.argsort(abs(tr['x'][:,j]-np.median(tr['x'][:,j])))[-20:] for j in range(4)]
for name,(a,b) in models.items():
 results=[]
 for i,te in enumerate(partitions):
  ix=np.setdiff1d(np.arange(72),te);r,c,z=fit(F[ix],T[ix],t,a,b); H=-np.expm1(-t[:,None]*r); pred=np.column_stack((F[te][:,a]@c[:len(a)],F[te][:,b]@c[len(a):]))@H.T; e=abs(pred-T[te]); results.append([e.mean(),e.mean(0).max(),e[:,-1].mean(),z]);
  if i==0: print(name,'rates',r,'coeffs',c)
 print(name,'fixed',results[0],'worst',np.max(results,0),'mean',np.mean(results,0),'edges',np.array(results[-4:])[:,0])
