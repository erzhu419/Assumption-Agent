import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];x=tr['x'];y=tr['y'];T=tr['traj'];t=s['eval_times'];z=m._calibrate(s['panel_a'],x);c=m._branch(s['panel_c'],z,'v1','soft');d=m._branch(s['panel_c'],z,'v2','gauss');f=m._design(z,c,d)
from itertools import combinations
pairs=list(combinations(range(4),2)); R=np.column_stack([a[:,i]/(1+k*a[:,j]**2) for a in [c,d] for i,j in pairs for k in [.5,1.] ]+[np.tanh(a[:,i]+q*a[:,j]) for a in [z,c,d] for i,j in pairs for q in [-1,1]]+[z[:,i]*z[:,j]*z[:,k] for i,j,k in combinations(range(4),3)])
rng=np.random.default_rng(114); folds=[np.arange(72),*[rng.permutation(72) for _ in range(6)],np.argsort(np.linalg.norm(z,axis=1)),*[np.argsort(abs(z[:,i])) for i in range(4)]]
for index,idx in enumerate(folds):
 a,b=idx[:52],idx[52:]; F=f[a]; Y=T[a]; beta=np.linalg.lstsq(F,Y,rcond=None)[0]; Yh=F@beta
 def profile(p,joint):
  q=m._time(t,p); U=Yh if joint else Y
  return (U.T-q@np.linalg.lstsq(q,U.T,rcond=None)[0]).ravel()
 out=[]
 for joint in [False,True]:
  p=least_squares(profile,np.log([.4,1.8]),args=(joint,),bounds=(np.log([.12,1.]),np.log([.9,4.]))).x;q=m._time(t,p); P=np.c_[f[b]@np.linalg.lstsq(F,y[a],rcond=None)[0],f[b]@beta@np.linalg.pinv(q.T)@q.T];out.append((np.exp(p),np.mean(abs(y[b]-P[:,0])),np.mean(abs(T[b]-P[:,1:])),max(np.mean(abs(T[b]-P[:,1:]),0))))
 fr=R-f@np.linalg.lstsq(F,R[a],rcond=None)[0]; fr/=np.maximum(fr[a].std(0),1e-6); A=fr[a]; yy=np.c_[y,T]; base=f@np.linalg.lstsq(F,yy[a],rcond=None)[0];
 for reg in [30,300]:
  pp=base[b]+fr[b]@np.linalg.solve(A.T@A+reg*np.eye(A.shape[1]),A.T@(yy[a]-base[a])); out.append((reg,np.mean(abs(y[b]-pp[:,0])),np.mean(abs(T[b]-pp[:,1:])),max(np.mean(abs(T[b]-pp[:,1:]),0))))
 print(index,out)
print('fields',list(s),'samples',len(x),'times',t)
print('calibrated coordinate range train/test',np.max(abs(z),0),np.max(abs(m._calibrate(s['panel_a'],s['test']['x'])),0))
