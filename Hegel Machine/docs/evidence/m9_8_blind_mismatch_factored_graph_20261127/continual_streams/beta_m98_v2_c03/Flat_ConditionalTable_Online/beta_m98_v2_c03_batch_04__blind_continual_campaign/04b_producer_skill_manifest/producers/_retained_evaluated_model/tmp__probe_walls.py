exec(open('/tmp/probe_m98.py').read().split('for a,b in product')[0])
import json
from scipy.optimize import least_squares
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print('apparatus', {k:v for k,v in e.items() if k!='facility_telemetry_trace'})
print('study scalars', {k:v for k,v in s.items() if np.isscalar(v)}); print('sensors',sensors.tolist())
print('row banks',np.unique(s['row_bank_id'].reshape(-1,4),axis=0))

def calc(par,fold=-1):
 axes=[]
 for j,coord in enumerate('xy'):
  k,v,h=s['kappa_'+coord],s['vel_'+coord],s['L'+coord]/(s['n'+coord]-1)
  A=mods['01']._axis(k,v,h).tolil(); l,r=par[j*2:j*2+2]; A[0,0]+=l/h; A[-1,-1]+=r/h; axes.append(A.tocsc())
 L=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); S=np.vstack([splu(L+mu*I).solve(src)[obs] for mu in s['mu_shifts']]); train=groups!=fold
 zz=Z[train]; D=np.c_[S[train],yc[train]]; D-=zz@np.linalg.solve(zz.T@zz+zr*np.eye(Z.shape[1]),zz.T@D); X,t=D[:,:-1],D[:,-1]; w,nu=opt(X.T@X+ridge*np.eye(X.shape[1]),-X.T@t,cap,budget); g=np.linalg.solve(zz.T@zz+zr*np.eye(Z.shape[1]),zz.T@(yc[train]-S[train]@w)); return S@w+Z@g-yc,w,g
b=np.repeat([s['robin_x'],s['robin_y']],2); print('baseline',b)
r,w,g=calc(b); print('baseline weights',w,'gamma',g); print('baseline residual per rosette',r.reshape(-1,4).round(5));
for width in (1,2,4):
 def expand(q): return np.repeat(q,4//width)
 for fold in range(-1,len(s['mu_shifts'])):
  train=groups!=fold; test=~train if fold>=0 else train
  fit=least_squares(lambda q:calc(expand(q),fold)[0][train],b[::4//width],bounds=(0,10),diff_step=1e-4,ftol=1e-10,gtol=1e-10); r,w,g=calc(expand(fit.x),fold); print('width/fold',width,fold,'fit',fit.x.round(6),'baseline/fit RMS',np.sqrt(np.mean(calc(b,fold)[0][test]**2)),np.sqrt(np.mean(r[test]**2)),'gamma',g.round(4))
