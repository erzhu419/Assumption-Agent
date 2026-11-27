import pickle
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
s,g=(lambda d:(d['study'],d['gt']))(pickle.load(open('public_beta_m96_w01_s00.pkl','rb'))[0])
z=s['Z_nuisance']; q,d,_=np.linalg.svd(z,full_matrices=False); ridge=s['nuisance_ridge']
profiles={'residual':d*d/(d*d+ridge),'sqrt':1-np.sqrt(ridge/(d*d+ridge)),'ols':np.ones_like(d),'none':np.zeros_like(d)}
def prof(a,t): return a-q@((t[:,None] if a.ndim==2 else t)*(q.T@a))
cs=s['sensor_calibration_sweeps']; bank=s['row_bank_id']; y=s['y_obs_raw'].copy()
for b in range(4):
 j=cs['bank_id']==b; x=cs['ref_signal'][j]; yy=cs['raw_readout'][j]; p=np.polyfit(x,yy,1); print('sweep',b,len(x),p,np.max(np.abs(yy-np.polyval(p,x)))); y[bank==b]=(y[bank==b]-p[1])/p[0]
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
print('response profiles',[(p,rel(prof(y,t),g['r_y'])) for p,t in profiles.items()])
rx=s['reaction_x'][:,None]+s['reaction_y'][None,:]; v=s['V_reaction_2d']; additive=v.mean(axis=1,keepdims=True)+v.mean(axis=0,keepdims=True)-v.mean()
print('reaction max difference',np.max(abs(v-rx)), 'nonadditivity',np.max(abs(v-additive)))
def op(axis,kind):
 n=s['n'+axis]; h=s['L'+axis]/n; k=s['kappa_'+axis]; v=s['vel_'+axis]/h; r=s['robin_'+axis]; c=2*k[1:]*k[:-1]/(k[1:]+k[:-1])/h**2
 dd=np.r_[0,c]+np.r_[c,0]; dd[[0,-1]]+=r*k[[0,-1]]/(h*(r*h/2+k[[0,-1]])); lo=-c.copy(); hi=-c.copy()
 if kind in ('cell','cell_drop_inflow'):
  dv=np.abs(v); lo-=np.maximum(v[1:],0); hi-=np.maximum(-v[:-1],0)
  if kind=='cell_drop_inflow': dv[0]-=max(v[0],0); dv[-1]-=max(-v[-1],0)
 elif kind in ('flux','flux_closed','flux_interior_diag','face_advective'):
  f=(v[:-1]+v[1:])/2; lo-=np.maximum(f,0); hi-=np.maximum(-f,0)
  if kind=='face_advective': dv=np.r_[max(v[0],0),np.maximum(f,0)]+np.r_[np.maximum(-f,0),max(-v[-1],0)]
  else:
   dv=np.r_[np.maximum(f,0),0]+np.r_[0,np.maximum(-f,0)]
   if kind=='flux': dv[[0,-1]] += [max(-v[0],0),max(v[-1],0)]
   if kind=='flux_interior_diag': dv[[0,-1]] += [abs(v[0]),abs(v[-1])]
 else: dv=0
 return diags([lo,dd+dv,hi],[-1,0,1],format='csc')
nx,ny=s['nx'],s['ny']; N=nx*ny; eyeN=eye(N,format='csc'); src=s['source_modes'].reshape(-1,N).T.copy(); i,j=s['sensor_ij'].T; ix=i*ny+j
kinds=['cell','cell_drop_inflow','flux','flux_closed','flux_interior_diag','face_advective','none']; bases={a:{k:op(a,k) for k in kinds} for a in 'xy'}; records=[]; terminal=[]
for k in kinds:
 A=kron(bases['x'][k],eye(ny),format='csc')+kron(eye(nx),bases['y'][k],format='csc')
 for name,R in [('axis',rx),('full',v),('additive',additive),('both',v+rx),('zero',np.zeros_like(v))]:
  B=A+diags(R.ravel(),format='csc'); u=np.stack([splu(B+mu*eyeN).solve(src) for mu in s['mu_shifts']]); pred=u[-1]@g['w_opt']; terminal.append((rel(pred,g['terminal_field_A'].ravel()),k,name))
  meas=u[:,ix,:];
  for layout,arr in [('shift_first',meas),('sensor_first',meas.transpose(1,0,2))]:
   D=arr.reshape(-1,src.shape[1]); records.extend((rel(prof(D,t),g['S_tilde']),k,name,layout,p) for p,t in profiles.items())
print('design best',*sorted(records)[:10],sep='\n'); print('terminal best',*sorted(terminal)[:10],sep='\n')
