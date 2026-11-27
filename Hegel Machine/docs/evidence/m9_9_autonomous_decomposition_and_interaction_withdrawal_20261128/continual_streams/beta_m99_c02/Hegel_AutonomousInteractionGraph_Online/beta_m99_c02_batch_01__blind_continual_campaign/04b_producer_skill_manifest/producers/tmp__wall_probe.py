import importlib.util, pickle, json
import numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/beta_m99_c02_s00_solver.py'
spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=a['study']; pulses=a['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T
Y=np.column_stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); print('cal pulse range and std',Y.min(),Y.max(),Y.std()); print('aux telemetry means',s['aux_manifold_telemetry'].mean(0))
for d in 'xy': print(d, 'edge kappa',s['kappa_'+d][[0,1,-2,-1]],'edge velocity',s['vel_'+d][[0,1,-2,-1]])
def response(par, ties=False):
 axes=[]
 for i,d in enumerate('xy'):
  k=s['kappa_'+d]; n=len(k); h=s['L'+d]/(n-1); r=s['robin_'+d]; A=m._axis(k,s['vel_'+d],h,r).tolil()
  for j,(row,near,wrap) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
   v,leak,cross=par[i if ties else 2*i+j]; rate=k[row]/h**2
   A[row,row]=v*rate+leak*r/h+cross*rate; A[row,near]=-v*rate; A[row,wrap]=-cross*rate
  axes.append(A.tocsc())
 return m._responses(s,axes)[0]@W

def report(name,P): print(name,'pulse/shift RMS',np.sqrt(np.mean((P-Y).reshape(3,-1,3)**2,axis=1)).round(8).tolist())
for name,row in [('open',[1,1,0]),('reflecting',[1,0,0]),('reflected ghost',[2,0,0]),('periodic',[1,0,1]),('x closed only',None),('y closed only',None)]:
 pp=np.tile(row,(2,1)) if row else [[1,0,0],[1,1,0]] if name.startswith('x') else [[1,1,0],[1,0,0]]
 report(name,response(pp,True))
for ties in [True,False]:
 n=2 if ties else 4
 for held in [-1,0,1,2]:
  mask=np.arange(3)!=held
  sol=least_squares(lambda x:(response(x.reshape(n,3),ties)-Y)[:,mask].ravel(),np.tile([1,.1,.01],n),bounds=(np.tile([.05,0,0],n),np.tile([4,4,4],n)),gtol=1e-13,ftol=1e-13,xtol=1e-13)
  print('fit',ties,'held',held,'edge [near,Robin,wrap] factors',sol.x.reshape(n,3).round(6).tolist(),'jac singular range',np.linalg.svd(sol.jac,compute_uv=False)[[0,-1]])
  report('fit',response(sol.x.reshape(n,3),ties))
