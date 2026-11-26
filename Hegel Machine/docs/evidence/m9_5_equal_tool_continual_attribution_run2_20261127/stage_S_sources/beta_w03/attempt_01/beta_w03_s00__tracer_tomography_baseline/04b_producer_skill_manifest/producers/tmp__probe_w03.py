import itertools,pickle
import numpy as np
from scipy.linalg import eig
s,g= (lambda d:(d['study'],d['gt']))(pickle.load(open('/holdout_task/public_beta_w03_s00.pkl','rb'))[0])
def build(L,n,k,v,r,diag,kind):
 h=L/n; d=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2
 a=np.diag(np.r_[0,d]+np.r_[d,0]+diag)+np.diag(-d,1)+np.diag(-d,-1)
 b=np.array([0,n-1]); a[b,b]+=r*k[b]/(h*(r*h*.5+k[b]))
 p=np.maximum(v,0)/h; q=np.maximum(-v,0)/h
 if kind.startswith('cell'):
  a+=np.diag(p+q)-np.diag(p[1:],-1)-np.diag(q[:-1],1)
  if kind=='cell_noin': a[0,0]-=p[0]; a[-1,-1]-=q[-1]
  if kind=='cell_doublein': a[0,0]+=p[0]; a[-1,-1]+=q[-1]
 else:
  f=(v[:-1]+v[1:])/(2*h); fp=np.maximum(f,0); fn=np.maximum(-f,0)
  a+=np.diag(np.r_[fp,0]+np.r_[0,fn])-np.diag(fp,-1)-np.diag(fn,1)
  if kind=='flux_open': a[0,0]+=q[0]; a[-1,-1]+=p[-1]
  if kind=='flux_cell': a[0,0]+=(q[0]-p[0]); a[-1,-1]+=(p[-1]-q[-1])
 return a
def fields(a,b):
 ex,px=eig(a); ey,py=eig(b)
 m=np.linalg.solve(px,s['source_modes'])@np.linalg.inv(py).T
 mid=m[None,:,:,:]/(ex[None,None,:,None]+ey[None,None,None,:]+s['mu_shifts'][:,None,None,None]); return (px@mid@py.T).real
Z=s['Z_nuisance']; z,sv,_=np.linalg.svd(Z,full_matrices=False); l=s['nuisance_ridge']; P=lambda Y,p: Y-z@((1-(l/(sv*sv+l))**p).reshape((-1,)+(1,)*(Y.ndim-1))*(z.T@Y))
y=s['y_obs_raw']; G=g['S_tilde']; R=g['r_y']; V=s['V_reaction_2d']; print('separable max defect',abs(V-V.mean(1)[:,None]-V.mean(0)[None,:]+V.mean()).max());print('target norms',np.linalg.norm(G),np.linalg.norm(R),'target head',G[:3,:2],R[:3]); print('target y residual options',[(p,np.linalg.norm(P(y,p)-R)) for p in [0,.5,1]],'sensor range',s['sensor_ij'].min(0),s['sensor_ij'].max(0),'bank rows',s['row_bank_id'][:44]);print('reference cert',g['cert_opt'])
for p in [0,.5,1]: print('y error',p,(P(y,p)-R)[:12])
fits=[]
for kind,react in itertools.product(['cell','cell_noin','cell_doublein','flux_closed','flux_open','flux_cell'],[0,1]):
 a,b=[build(s['L'+d],s['n'+d],s['kappa_'+d],s['vel_'+d],s['robin_'+d],v+react*s['reaction_'+d],kind) for d,v in zip('xy',[V.mean(axis=1),V.mean(axis=0)-V.mean()])]
 U=fields(a,b); field=g['w_opt']@U[-1].reshape(12,-1); ferr=np.linalg.norm(field-g['terminal_field_A'].ravel())/np.linalg.norm(g['terminal_field_A']);
 for off,order,p in itertools.product([0,1],['shift-major','sensor-major'],[0,.5,1]):
  i,j=(s['sensor_ij']-off).T; X=U[:,:,i,j]; S=(X.transpose(0,2,1) if order=='shift-major' else X.transpose(2,0,1)).reshape(-1,12)
  err=np.linalg.norm(P(S,p)-G)/np.linalg.norm(G);fits.append((err,ferr,kind,react,off,order,p))
print('best design',*sorted(fits)[:10],sep='\n');print('field errors',sorted(set((r[1],r[2],r[3]) for r in fits)))
