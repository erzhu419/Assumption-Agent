import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c03_s00_solver as a
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; D=item['campaign_diagnostics']; W=np.array([d['w_pulse'] for d in D]).T; Y=np.stack([a._calibrate(s,d['y_pulse_raw']) for d in D],axis=1)
def axes(p):
 out=[]
 for d,n,P in zip('xy',(s['nx'],s['ny']),np.reshape(p,(2,2,2))):
  h=s['L'+d]/(n-1); k=s['kappa_'+d]; r=s['robin_'+d]; A=a._axis(k,s['vel_'+d],h,r).tolil()
  for i,j,(c,b) in zip((0,n-1),(1,n-2),P):
   A[i,i]=c*k[i]/h**2+b*r/h; A[i,j]=-c*k[i]/h**2
  out.append(A.tocsc())
 return out
def pred(p): return a._responses(s,axes(p))[0]@W
print('shifts',s['mu_shifts'],'h',s['Lx']/(s['nx']-1),s['Ly']/(s['ny']-1)); print('kap endpoints',s['kappa_x'][[0,-1]],s['kappa_y'][[0,-1]]); print('vel endpoints',s['vel_x'][[0,-1]],s['vel_y'][[0,-1]])
for hold in [-1,0,1,2]:
 tr=np.arange(3)!=hold
 for mode in [0,1,2,3]:
  def expand(p):
   if mode==0: return np.tile(p,4)
   if mode==1: return np.repeat(np.reshape(p,(2,2)),2,axis=0).ravel()
   if mode==2: return np.column_stack((np.ones(4),p)).ravel()
   return p
  n=[2,4,4,8][mode]; fit=least_squares(lambda p:(pred(expand(p))[:,tr]-Y[:,tr]).ravel(),np.ones(n),bounds=(0,np.inf),gtol=2e-13,ftol=2e-13,xtol=2e-13)
  err=pred(expand(fit.x))-Y; print('hold',hold,'mode',mode,'coeffs',fit.x.round(7),'RMSE',np.sqrt(np.mean(err.reshape(3,32,3)**2,axis=1)).round(8).tolist())
