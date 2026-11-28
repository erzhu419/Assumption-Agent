import numpy as np,pickle,sys,runpy
from scipy.optimize import minimize_scalar
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; arc=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py'); pred=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; u,v=(S['panel_b'][k] for k in ('u','v')); n=len(S['train']['x']); X=arc['_operator'](dict(S,panel_b={'u':np.r_[np.zeros((1,4)),np.eye(4)],'v':np.r_[np.zeros((1,4)),np.eye(4)]}),S['train']['x']); aff=arc['_aff']; ls=arc['_ls']; print('x ranges',u.min(0),u.max(0),X.min(0),X.max(0))
def cv(W,masks):
 r=[]
 for test in masks:
  train=np.setdiff1d(np.arange(n),test); o=pred({'__input__':{'train':{k:S['train'][k][train] for k in ('y','traj')}},'node_00':{'w':W[np.r_[train,test]]}}); r.append([np.mean(abs(o['y_hat']-S['train']['y'][test])),np.mean(abs(o['state_pred']-S['train']['traj'][test]))])
 return np.array(r)
masks=[np.arange(52,72),*np.array_split(np.random.default_rng(21).permutation(n),4)]
funcs={'power':(lambda z,g:z*abs(z)**(g-1),1.25,(1,6)), 'powerlin':(lambda z,g:z*abs(z)**(g-1),3,(1.1,6)), 'tanh':(lambda z,g:np.tanh(g*z),1.25,(.15,5)), 'sin':(lambda z,g:np.sin(g*z),1.25,(.15,5)), 'sinh':(lambda z,g:np.sinh(g*z),1.25,(.05,5)), 'asinh':(lambda z,g:np.arcsinh(g*z),1.25,(.05,5)), 'atan':(lambda z,g:np.arctan(g*z),1.25,(.05,5)), 'rat':(lambda z,g:z**3/(1+g*z*z),1.25,(0,8)), 'tanhcube':(lambda z,g:np.tanh(g*z)**3,1.25,(.05,5)), 'cubelin':(lambda z,g:z**3+g*z,.22,(-1,3)), 'cubelinInvG':(lambda z,g:z+g*z**3,1.25,(0,50)), 'sqrtcube':(lambda z,g:z**3/np.sqrt(1+g*z*z),1.25,(0,20)), 'coshcube':(lambda z,g:z**3/np.cosh(g*z),1.25,(0,4)), 'quant':(lambda z,g:z+g*np.sign(z),1.25,(-1,2)), 'rat2':(lambda z,g:z**3/(1+g*abs(z)),1.25,(0,20)), 'inv3':(lambda z,g:np.sinh(np.arcsinh(g*z)/3),1.25,(.05,8)), 'softsign':(lambda z,g:z/(1+g*abs(z)),1.25,(.05,8))}
results=[]
def check(name,F):
 A=F(u); b=ls(A,v); err=A@b-v; h=(np.linalg.svd(A,full_matrices=False)[0]**2).sum(1); fit=np.sqrt(np.mean(err**2)); loo=np.sqrt(np.mean((err/(1-h)[:,None])**2)); C=cv(F(X)@b,masks); results.append((name,fit,loo,*C[0],*C[1:].max(0)))
check('affine',aff)
for name,(fn,g0,rg) in funcs.items():
 for plus in (False,True):
  def F(z,g): return np.c_[aff(z),fn(z,g)] if plus else aff(fn(z,g))
  def obj(g):
   A=F(u,g);return np.mean((A@ls(A,v)-v)**2)
  G=np.linspace(*rg,95); i=np.argmin([obj(g) for g in G]); g=minimize_scalar(obj,bounds=(G[max(i-1,0)],G[min(i+1,len(G)-1)]),method='bounded').x
  for g in set([g0,g,1.25] if not name.startswith('power') else [g0,g,3.]): check(f'{name}+{int(plus)}@{g:.6f}',lambda z:F(z,g))
print('name, panel fit, panel LOO, public y/traj, worst 4fold y/traj');
for strs,vals in [(r[0],r[1:]) for r in sorted(results,key=lambda r:r[2])[:25]]: print(strs,np.round(vals,6))
print('best downstream')
for strs,vals in [(r[0],r[1:]) for r in sorted(results,key=lambda r:max(r[3:5]))[:20]]: print(strs,np.round(vals,6))
