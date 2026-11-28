import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _calibrate,_operators,_aff,_ls,_time_basis
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=52
p,r=_operators(s['panel_c'],_calibrate(s['panel_a'],tr['x'])); traj=tr['traj']; a=_time_basis(s['eval_times'],traj[:n]); z=_ls(a,traj.T).T
dot=(p*r).sum(1); h=_aff(p,r,dot)
cross=np.column_stack([p[:,i]*r[:,j] for i,j in itertools.product(range(4),repeat=2) if i!=j]); labels=[f'p{i}r{j}' for i,j in itertools.product(range(4),repeat=2) if i!=j]
for j in range(2):
 for name,base in [('p',_aff(p,dot)),('r',_aff(r,dot)),('pr',h),('r_no_dot',_aff(r))]:
  best=[]
  for k in range(3):
   for ids in itertools.combinations(range(len(labels)),k):
    X=np.c_[base,cross[:,ids]]; w=_ls(X[:n],z[:n,j]); e=X@w-z[:,j]; best.append((np.sqrt(np.mean(e[:n]**2)),np.mean(abs(e[n:])),[labels[i] for i in ids],w))
  print(j,name,sorted(best,key=lambda T:T[0])[:3])
print('slow coefficients no additional terms',_ls(h[:n],z[:n,1]))
