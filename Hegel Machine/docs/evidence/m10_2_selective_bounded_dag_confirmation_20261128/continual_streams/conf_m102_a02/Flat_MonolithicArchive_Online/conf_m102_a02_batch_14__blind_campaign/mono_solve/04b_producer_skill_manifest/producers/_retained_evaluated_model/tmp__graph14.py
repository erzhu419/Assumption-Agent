from panels14 import *
import importlib.util,itertools
p=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/conf_m102_a02_batch_09_solver.py'); m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
def calibrate(s):return m._calibrate(s['panel_a'])
def panel(u,v,terms):
 def D(x,p):return np.c_[aff(x),*[F[k](x,a) for k,a in zip(terms,p)]]
 def err(p):
  d=D(u,p);return (d@ls(d,v)-v).ravel()
 init={'sin':2.15,'cos':1.25,'rat':1.,'tanh':1.25,'gauss':1.,'soft':.83,'exp':.3,'rsq':1.}
 fit=least_squares(err,[init[a] for a in terms],bounds=(.02,8),max_nfev=90)
 w=ls(D(u,fit.x),v)
 return lambda x:D(x,fit.x)@w,fit.x,w,np.std(fit.fun)
def latent(s,x):
 z=m._inverse(x,calibrate(s)); B=s['panel_b']; C=s['panel_c'];
 b=panel(B['u'],B['v'],['gauss'])[0](z); c=panel(C['u'],C['v1'],['soft'])[0](z); d=panel(C['u'],C['v2'],['exp'])[0](z)
 return dict(x=x,z=z,b=b,c=c,d=d)
def feats(a,deg=2):
 i,j=np.triu_indices(a.shape[1]);return np.c_[aff(a),a[:,i]*a[:,j]] if deg==2 else aff(a)
if __name__=='__main__':
 tr=S['train']; traj=tr['traj']; y=tr['y']; t=S['eval_times']; L=latent(S,tr['x']); Y=np.c_[y,traj]; B=S['panel_b']; C=S['panel_c']
 for u,v in [(B['u'],B['v']),(C['u'],C['v1']),(C['u'],C['v2'])]:
  outs=[]
  for terms in itertools.combinations(['sin','cos','tanh','rat','gauss','soft','exp','rsq'],2):
   _,p,w,e=panel(u,v,terms);outs.append((e,terms,p,np.linalg.norm(w,axis=0)))
  print('operator pairs',sorted(outs,key=lambda x:x[0])[:7]);
  for term in ['sin','tanh','rat','gauss','soft','exp']:
   _,p,w,e=panel(u,v,[term]); print(term,p,e,np.linalg.svd(w[5:],compute_uv=False), np.round(w,3) if (term=='gauss' and u is B['u']) or (term in ['soft','exp'] and u is C['u']) else '')
 for r in range(1,4):
  def tm(p):return -np.expm1(-t[:,None]*p)
  def fun(p):d=tm(p);return (d@ls(d,traj.T)-traj.T).ravel()
  fit=least_squares(fun,np.geomspace(.2,2,r),bounds=(.001,10),max_nfev=1000); print('time',fit.x,np.std(fit.fun))
 for deg in [1,2]:
  out=[]
  for n in range(1,4):
   for keys in itertools.combinations(L,n):
    a=feats(np.column_stack([L[k] for k in keys]),deg); aa=a[:52]; aq=a[52:];
    for la in ([0] if deg==1 else [0,.01,.1,1.]):
     w=ls(np.r_[aa,np.sqrt(la)*np.eye(a.shape[1])],np.r_[Y[:52],np.zeros((a.shape[1],9))]); E=abs(aq@w-Y[52:]);out.append((E[:,0].mean(),E[:,1:].mean(),keys,la))
  print('baseline',deg,'scalar',sorted(out)[:12]);print('trajectory',sorted(out,key=lambda r:r[1])[:12])
 print('ranges',{k:(v.min(),v.max()) for k,v in L.items()})
