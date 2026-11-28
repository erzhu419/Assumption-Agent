from probe import *
from scipy.stats import pearsonr
x,y,z=[S['train'][k] for k in ('x','y','traj')];t=S['eval_times']; n=len(x)
a=S['panel_a']; A,_=robust(design(a['u'],1),a['v']); b=S['panel_b'];B=np.linalg.lstsq(design(b['u'],1),b['v'],rcond=None)[0]
def af(x,M): return x@M[1:]+M[0]
spaces={'x':x,'a':af(x,A),'ai':(x-A[0])@np.linalg.inv(A[1:]),'b':af(x,B),'bi':(x-B[0])@np.linalg.inv(B[1:]),'ba':af(af(x,A),B),'bai':af((x-A[0])@np.linalg.inv(A[1:]),B)}
for key,v in spaces.items():
 P=design(v,2); w=np.linalg.lstsq(P,np.c_[y,z[:,-1]],rcond=None)[0]; print(key,'linear and quadratic coefficients',np.round(w.T,3))
for rate in [.65,.7,.72,.75,.8,.85,1.0]:
 V=-np.expm1(-t*rate); C=z@V/(V@V);print('single time',rate,np.sqrt(np.mean((z-C[:,None]*V)**2)))
V=-np.expm1(-t[:,None]*np.array([.725,4.085]));C=np.linalg.lstsq(V,z.T,rcond=None)[0].T
print('time coefficients std and correlations',C.std(0),np.corrcoef(np.c_[C,y].T))
for j in range(2):
 w=np.linalg.lstsq(design(x,2),C[:,j],rcond=None)[0];print('amp',j,'quadratic',np.round(w,4),'rms',np.std(C[:,j]-design(x,2)@w))
