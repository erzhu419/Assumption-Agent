import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
from scipy.optimize import least_squares
from scipy.linalg import svd
for panel,key in [('panel_b','v'),('panel_c','v1')]:
 U=s[panel]['u'];V=s[panel][key]; rows=[]
 for name,F in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('soft',lambda a:a/np.sqrt(1+a*a)),('rat',lambda a:a/(1+a*a)),('erf',__import__('scipy').special.erf),('gauss',lambda a:a*np.exp(-a*a)),('cube',lambda a:a**3)]:
  for mode in [0,1,2]:
   def D(X,p):
    if mode==0:return f._affine(X+p[1]*F(p[0]*X))
    if mode==1:return f._affine(X+p[1]*F(p[0]*(X+p[2])))
    return f._affine(X+p[1]*F(p[0]*(X+p[2:6])))
   def loss(p):
    A=D(U,p);return (A@ls(A,V)-V).ravel()
   best=min([least_squares(loss,[h,b]+([0]*([0,1,4][mode])),bounds=([.08,-20]+[-2]*([0,1,4][mode]),[8,20]+[2]*([0,1,4][mode])),max_nfev=180) for h in [.7,1.5,2.8] for b in [-.7,.5]], key=lambda q:q.cost)
   A=D(U,best.x);C=ls(A,V);rows.append((np.sqrt(best.cost*2),name,mode,best.x,D(raw,best.x)@C))
 for e,n,m,p,X in sorted(rows,key=lambda q:q[0])[:7]:
  print(panel,n,m,round(e,6),p.round(5));
  if panel!='panel_b':continue
  D,terms=poly(X,2); co=ls(D[:52],np.c_[y,T][:52]);E=abs(D@co-np.c_[y,T]); print('train',E[:52].mean(0).round(4),'val',E[52:].mean(0).round(4));
  if e==min(a[0] for a in rows):
   print('range',X.min(0),X.max(0)); print('yco',list(zip(terms,co[:,0].round(4))));print('trco',co[:,[1,-1]].round(4));print('corr T y',np.corrcoef(T[:,-1],y)[0,1]);np.savez('/tmp/aligned.npz',X=X,raw=raw,y=y,T=T,t=t)
print('keys',s.keys(), 'shapes',u.shape, y.shape);print('sensor',f._fit_sensor(s['panel_a']));
