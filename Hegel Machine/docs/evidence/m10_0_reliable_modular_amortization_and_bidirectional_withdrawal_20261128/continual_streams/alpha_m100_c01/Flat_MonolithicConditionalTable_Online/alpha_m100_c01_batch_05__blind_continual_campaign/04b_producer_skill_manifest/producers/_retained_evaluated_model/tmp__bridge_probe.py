import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; clean=np.max(abs(Y),axis=1)<5; X,Y=X[clean],Y[clean]; ids=np.flatnonzero(clean)
F={'tanh':np.tanh,'atan':np.arctan,'sqrt':lambda z:z/np.sqrt(1+z*z),'rational':lambda z:z/(1+z*z),'softsign':lambda z:z/(1+abs(z)),'exp':lambda z:np.sign(z)*-np.expm1(-abs(z))}
# Separate held-out sets for mixed and isolated sweeps.
fit=(ids%4!=0); fit|=(ids<16)
for j in range(4):
 x=X[:,j]; z=X[:,(j+2)%4]; u=X[:,(j+1)%4]; v=X[:,(j-1)%4]; y=Y[:,j]
 for name,f in F.items():
  for variant in (0,1,2,3):
   terms=[np.ones(len(x)),np.maximum(z,0),np.minimum(z,0),x,z*z,x*z]
   if variant==1: terms=terms[:4]
   if variant==2: terms += [u,v,x*u,x*v,u*v]
   if variant==3: terms += [abs(x),u,v,x*u,x*v,u*v,x*x,x*z*z,x*x*z]
   M=np.array(terms).T
   def pred(p): return M@p[3:]+p[0]*f(x*np.exp(np.where(x>=0,p[1],p[2])))
   r=least_squares(lambda p:(pred(p)-y)[fit],np.r_[1,0,0,np.zeros(M.shape[1])],loss='soft_l1',f_scale=.005,max_nfev=250,bounds=(np.r_[0,-3,-3,np.full(M.shape[1],-np.inf)],np.r_[10,4,4,np.full(M.shape[1],np.inf)])); e=pred(r.x)-y
   print(j,name,variant,np.round([np.sqrt(np.mean(e[fit]**2)),np.sqrt(np.mean(e[~fit]**2)),max(abs(e[~fit]))],6),np.round(r.x,4) if name=='tanh' or max(abs(e[~fit]))<.002 else '')
