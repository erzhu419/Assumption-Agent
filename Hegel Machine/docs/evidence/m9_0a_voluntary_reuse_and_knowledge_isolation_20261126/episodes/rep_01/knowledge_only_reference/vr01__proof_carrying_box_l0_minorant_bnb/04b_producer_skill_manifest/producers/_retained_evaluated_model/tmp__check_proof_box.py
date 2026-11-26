import importlib.util,pickle,time
import numpy as np
spec=importlib.util.spec_from_file_location('proof_box_l0','/holdout_task/proof_box_l0.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def check(q,s):
 X,y,M,l0,l2=(q[k] for k in ('X','y','M','lam0','lam2')); w=s['weights'];ub=s['objective'];lb=s['certified_lower_bound'];p=X.shape[1]; leaves=s['subcube_leaves']; G=X.T@X
 assert np.all(np.isfinite(w)) and w.shape==(p,) and max(abs(w))<=M+1e-12
 assert abs(ub-(.5*np.linalg.norm(X@w-y)**2+.5*l2*(w@w)+l0*sum(abs(w)>1e-8)))<1e-9
 assert 1<=len(leaves)<=2048 and 0<=ub-lb<=1e-5
 cubes=[]; vol=0; max_over=-np.inf; min_eig=np.inf
 for leaf in leaves:
  z,a=(np.asarray(leaf[k]) for k in ('fixed_zero','fixed_active')); assert a.shape==z.shape==(p,) and not np.any(z&a)
  if 'dual_nu' in leaf:
   nu=leaf['dual_nu']; cst=nu@y-.5*(nu@nu); h=abs(X.T@nu); qq=l2
  else:
   d,c=leaf['minorant_diagonal'],leaf['minorant_center']; e=np.linalg.eigvalsh(G-np.diag(d))[0]; min_eig=min(min_eig,e); assert min(d)>=0 and e>=-1e-10
   r=y-X@c; cst=.5*(r@r)+r@(X@c)+.5*((d*c)@c); h=abs(X.T@r+d*c); qq=l2+d
  b=np.minimum(M,h/qq); on=l0-b*(h-.5*qq*b); val=cst+on[a].sum()+np.minimum(0,on[~(a|z)]).sum()
  max_over=max(max_over,leaf['lb']-val); assert leaf['lb']<=val+1e-9 and lb<=leaf['lb']+1e-6
  zi=sum(1<<int(j) for j in np.flatnonzero(z)); ai=sum(1<<int(j) for j in np.flatnonzero(a))
  assert all((zi&aj) or (ai&zj) for zj,aj in cubes); cubes.append((zi,ai)); vol+=1<<(p-int(sum(z|a)))
 assert vol==1<<p
 return ub-lb,len(leaves),min_eig,max_over
if __name__=='__main__':
 for row in pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb')):
  q=row['problem']; gt=row['gt']; start=time.monotonic(); s=m.solve_proof_carrying_box_l0(q); w=gt['exact_weights']; print(q['X'].shape,check(q,s),'objective',s['objective'],'reference',gt['exact_objective'],'reference residual objective',.5*np.linalg.norm(q['X']@w-q['y'])**2+.5*q['lam2']*(w@w)+q['lam0']*sum(abs(w)>1e-8),'secs',time.monotonic()-start,flush=True)
