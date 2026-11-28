import sys,pickle,numpy as np,runpy,itertools
sys.path.insert(0,'/holdout_task'); import node_00 as n
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; q=n._inverse_a(tr['x'],n._fit_panel_a(s['panel_a'])); b=n._fit_panel_b(s['panel_b']); M=b[1:]; y=tr['y']; t=tr['traj']; pairs=list(itertools.combinations_with_replacement(range(4),2)); aff=lambda x:np.column_stack((np.ones(len(x)),x)); Q=np.column_stack([q[:,i]*q[:,j] for i,j in pairs]); X=np.column_stack((aff(q),Q)); cv=np.arange(52); te=np.arange(52,72); c=np.linalg.lstsq(X[cv],t[cv],rcond=None)[0]; U,S,V=np.linalg.svd(c[5:],full_matrices=False); k=U[:,0]/U[1,0]; g=Q@k; C=np.linalg.lstsq(np.column_stack((aff(q),g))[cv],t[cv],rcond=None)[0]; d=np.linalg.lstsq(C[1:5].T,C[5],rcond=None)[0]; print('k',k,'d',d,'trajectory linear singulars',np.linalg.svd(C[1:5],compute_uv=False),'unexplained',C[5]-d@C[1:5]);
for shrink in [0,0.003,0.01,0.03,0.1]:
 L=C[1:5]; dd=np.linalg.solve(L@L.T+max(shrink,1e-8)**2*np.eye(4),L@C[5]); z=q+g[:,None]*dd; print('d',shrink,dd)
 for name,Z in [('q',z),('Bf',z@M+b[0]),('Bi',(z-b[0])@np.linalg.inv(M)),('Bft',z@M.T+b[0]),('Bit',(z-b[0])@np.linalg.inv(M.T))]:
  o=sink({'node_00':{'z':Z[cv],'z_test':Z[te],'y':y[cv],'traj':t[cv]}}); print(name,round(np.mean(abs(o['y_hat']-y[te])),5),round(np.mean(abs(o['state_pred']-t[te])),5))
# Fit a shared quadratic form to Panel C v2 after accounting for its separable fit.
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; v=p['v2'];
for name,func,lo,hi in [('cos',np.cos,.25,4),('sin',np.sin,.25,4)]:
 for mode in ['shared','free','shared_plus_quad']:
  def fit(f,ret=False):
   ff=lambda z:z+f[0]*func(f[1]*z)
   A=aff(ff(u)) if mode=='shared' else np.column_stack((aff(ff(u)),u*u)) if mode=='shared_plus_quad' else np.column_stack((aff(u),func(f[1]*u)))
   C=np.linalg.lstsq(A,v,rcond=None)[0]; r=A@C-v; return (f,C) if ret else r.ravel()
  from scipy.optimize import least_squares
  r=min((least_squares(fit,[a,w],bounds=([-3,lo],[3,hi])) for a in [-.5,.5] for w in [1.2,2.5,3.5]),key=lambda r:np.linalg.norm(r.fun));print('C2',name,mode,r.x,np.sqrt(np.mean(r.fun**2)))
print('telemetry',s['telemetry_vec'])
