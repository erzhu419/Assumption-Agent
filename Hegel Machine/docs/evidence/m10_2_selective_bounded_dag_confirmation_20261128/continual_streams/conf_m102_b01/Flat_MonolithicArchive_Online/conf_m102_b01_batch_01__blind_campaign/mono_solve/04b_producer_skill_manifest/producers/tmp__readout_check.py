import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];x=tr['x'];y=tr['y'];t=tr['traj'];z=b._latent(s,x);a=b._aff(z);d=b._scalar_design(z);ls=b._ls
ids=np.arange(len(x));rng=np.random.default_rng(517);folds=[ids]+[rng.permutation(ids) for _ in range(16)]+[j for j in np.argsort(abs(b._calibrate(s['panel_a'],x)),axis=0).T]
for mode in range(6):
 stats=[]
 for j in folds:
  i,k=j[:52],j[52:]; A=a if mode%2==0 else z; D=d if mode%2==0 else d[:,1:];coef=ls(A[i],t[i]);basis=t[i] if mode//2==1 else A[i]@coef
  if mode//2:V=np.linalg.svd(basis,full_matrices=False)[2][:2];coef=(coef@V.T)@V
  e=abs(A[k]@coef-t[k]);f=abs(D[k]@ls(D[i],y[i])-y[k]);stats.append([f.mean(),e.mean(),e.mean(0).max(),f.max(),e.max()]);
 print(mode,'52/20',np.round(stats[0],7),'tail',np.round(np.max(stats[-4:],axis=0),7),'worst',np.round(np.max(stats,axis=0),7))
