import sys,pickle,numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def axis(k,v,h,p):
 f=(k[1:]+k[:-1])/(2*h*h)
 lo=-f-np.maximum(v[1:],0)/h; hi=-f-np.maximum(-v[:-1],0)/h
 lo[-1]=-p[3]*k[-1]/h**2; hi[0]=-p[1]*k[0]/h**2
 d=np.r_[-hi[0]+p[0]/h,-lo[:-1]-hi[1:],-lo[-1]+p[2]/h]
 return diags([lo,d,hi],[-1,0,1],format='csc')

def response(s,p,rhs=None,ret=False):
 nx,ny=s['nx'],s['ny']; i,j=s['sensor_ij'].T; idx=i+nx*j
 ax,ay=[axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),p[4*k:4*k+4]) for k,a in enumerate('xy')]
 L=kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))
 if rhs is None: rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
 rows=[]
 for shift in s['mu_shifts']:
  U=splu((L+shift*eye(nx*ny)).tocsc()).solve(rhs); rows.append(U[idx])
 S=np.concatenate(rows)
 return (S,U.T.reshape(-1,ny,nx).transpose(0,2,1)) if ret else S

if __name__=='__main__':
 it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=calibrate_upstream(dict(it['study'],reference_tracer_pulses=it['campaign_diagnostics'])); ds=s['reference_tracer_pulses']; Y=np.array([d['y_pulse_raw'] for d in ds]).T; W=np.array([d['w_pulse'] for d in ds]).T
 base=np.array([s['robin_x'],1,s['robin_x'],1,s['robin_y'],1,s['robin_y'],1]); robin=np.arange(0,8,2); default=base.copy(); R=lambda p: response(s,p)@W-Y
 def display(name,p):
  e=R(p); print(name,np.round(p,7),'pulse RMSE',np.round(np.mean(e**2,axis=0)**.5,8),'max',abs(e).max())
 for x in [0,1]:
  for y in [0,1]:
   p=base.copy(); p[:4:2]*=x; p[4::2]*=y; display((x,y),p)
 for pat in ('axis','robin','all'):
  def expand(z):
   p=base.copy()
   if pat=='axis': p[robin]=np.repeat(z,2)
   elif pat=='robin': p[robin]=z
   else: p=z
   return p
  z0={'axis':base[::4],'robin':base[robin],'all':base}[pat]
  for held in (None,0,1,2):
   keep=[i for i in range(3) if i!=held]
   f=least_squares(lambda z:R(expand(z))[:,keep].ravel(),z0,bounds=(0,np.inf),gtol=1e-12,ftol=1e-12,xtol=1e-12)
   display((pat,held),expand(f.x))
 print('calibrated pulse range',Y.min(),Y.max())
