import pickle,numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
C=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; X,Y=C['true_channel_loads'],C['bridge_raw']; ok=np.max(abs(Y),1)<5; ids=np.nonzero(ok)[0]; X,Y=X[ok],Y[ok]; Z=np.roll(X,2,axis=1)
for n in (3,5,7,9,13):
 def basis(x): return BSpline.design_matrix(x,np.r_[[-1.8]*4,np.linspace(-1.8,1.8,n+2)[1:-1],[1.8]*4],3).toarray()
 errors=[]
 for j in range(4):
  a,b=basis(X[:,j]),basis(Z[:,j]); M=np.c_[a,b[:,1:]]; y=Y[:,j]; errs=[]
  for fold in range(4):
   keep=ids%4!=fold; p=np.linalg.lstsq(M[keep],y[keep],rcond=None)[0]; errs.extend((M[~keep]@p-y[~keep]).tolist())
  p=np.linalg.lstsq(M,y,rcond=None)[0]; errors.append([np.std(M@p-y),np.std(errs),max(abs(np.array(errs)))]);
  if n==9:
   grid=np.arange(-1.6,1.61,.1); base=basis(grid)-basis(np.array([0.])); own=base@p[:a.shape[1]]; cross=base[:,1:]@p[a.shape[1]:]; print('ch',j,'own',np.round(own,4),'cross',np.round(cross,4))
 print(n,np.round(errors,6))
