import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags
p='/tmp/hegel_continual_archive/beta_m98_v2_c04_batch_02_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); orig=m._axis
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; ds=item['campaign_diagnostics']; W=np.stack([d['w_pulse'] for d in ds],axis=1); Y=np.stack([m._calibrate({**s,'y_obs_raw':d['y_pulse_raw']}) for d in ds],axis=1)
Z=s['Z_nuisance']; print('shifts',s['mu_shifts'],'observation range',Y.min(),Y.max());
for a in 'xy':
 h=s['L'+a]/(s['n'+a]-1); k=s['kappa_'+a]; v=s['vel_'+a]; print(a,'local',k[[0,-1]]/h**2,'face',(k[[0,-1]]+k[[1,-2]])/(2*h**2),'velocity',v[[0,-1]]/h,'Robin',s['robin_'+a]/h)
def pred(p):
 def axis(k,v,L,r):
  h=L/(len(k)-1); q=p[:4] if k is s['kappa_x'] else p[4:]; A=orig(k,v,L,0).tolil(); A[0,1]=-q[0]*k[0]/h**2; A[-1,-2]=-q[1]*k[-1]/h**2; A[0,0]=-A[0,1]+q[2]*r/h; A[-1,-1]=-A[-1,-2]+q[3]*r/h; return A.tocsc()
 m._axis=axis
 return m._forward(s)[0]@W
for train,test in [([0,1],[0,1]),([0],[1]),([1],[0])]:
 def fun(p): return (pred(p)-Y)[:,train].ravel()
 fit=least_squares(fun,[1,1,.1,.1]*2,bounds=([0,0,-.2,-.2]*2,[8,8,8,8]*2),xtol=1e-12,gtol=1e-13,ftol=1e-12)
 e=pred(fit.x)-Y; print('train',train,'p',fit.x.round(7),'rmse-by-shift',np.sqrt(np.mean(e.reshape(3,-1,2)**2,axis=1)),'test rel',np.linalg.norm(e[:,test],axis=0)/np.linalg.norm(Y[:,test],axis=0),'jac sv',np.linalg.svd(fit.jac,compute_uv=False).round(5))
