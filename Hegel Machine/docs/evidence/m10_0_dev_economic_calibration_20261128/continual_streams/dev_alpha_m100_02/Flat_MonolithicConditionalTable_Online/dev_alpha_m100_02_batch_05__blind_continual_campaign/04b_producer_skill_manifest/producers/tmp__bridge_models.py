import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; S=C['single_channel_sweeps']; Q={j:np.column_stack((S['true_stress'][S['lab_id']==j],S['cmd_stress'][S['lab_id']==j])) for j in range(4)}
clean=(abs(Y)<5).all(1)
F={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'softsign':lambda z:z/np.sqrt(1+z*z),'erf':__import__('scipy.special',fromlist=['erf']).erf,'sine':np.sin}
for j,q in Q.items():
 x,y=q.T; k=abs(y)<5; x,y=x[k],y[k]; print('axis',j)
 for name,fn in F.items():
  for powers in ((1,),(3,),(1,3),()):
   def design(a): return np.column_stack([np.ones(len(x)),fn(a[0]*x)]+[x**n for n in powers])
   def err(a):
    m=design(a); return m@np.linalg.lstsq(m,y,rcond=None)[0]-y
   p=min([least_squares(err,[a],bounds=(.03,5.0),ftol=1e-12,gtol=1e-12,xtol=1e-12) for a in (.4,1.0,2.0)], key=lambda p:p.cost)
   if np.linalg.norm(p.fun)<.022: print(name,powers,'rmse',round(np.sqrt(np.mean(p.fun**2)),7),'param',np.r_[p.x,np.linalg.lstsq(design(p.x),y,rcond=None)[0]].round(5))

# Compare additive saturation with a saturation applied to a coupled load.
folds=np.array_split(np.random.default_rng(42).permutation(len(X)),5)
for name in ('tanh','atan','softsign'):
 f=F[name]
 for typ in ('post','pre','both'):
  for degree in (0,2,3):
   P=np.column_stack([np.prod(X[:,z],axis=1) for n in range(2,degree+1) for z in combinations_with_replacement(range(4),n)]) if degree else np.empty((len(X),0))
   out=np.zeros_like(Y)
   for j in range(4):
    def model(p):
     z=X[:,j]*p[2]; g=3
     if typ in ('pre','both'):
      z=z+X@p[g:g+4]*1.0+p[g+4]; g+=5
     phi=np.column_stack((X, X[:,j]**3, P)); return p[0]+p[1]*f(z)+phi@p[g:]
    start=np.r_[np.median(Y[clean,j]),1.0,1.2,np.zeros(5 if typ in ('pre','both') else 0),np.zeros(5+P.shape[1])]; start[-(5+P.shape[1])+j]=.2
    for ids in folds:
     keep=clean.copy(); keep[ids]=False
     opt=least_squares(lambda p:(model(p)-Y[:,j])[keep],start,max_nfev=450)
     out[ids,j]=model(opt.x)[ids]-Y[ids,j]
   print(name,typ,degree,'raw clean p50,90,99',np.quantile(abs(out[clean]),[.5,.9,.99]).round(6),flush=True)
