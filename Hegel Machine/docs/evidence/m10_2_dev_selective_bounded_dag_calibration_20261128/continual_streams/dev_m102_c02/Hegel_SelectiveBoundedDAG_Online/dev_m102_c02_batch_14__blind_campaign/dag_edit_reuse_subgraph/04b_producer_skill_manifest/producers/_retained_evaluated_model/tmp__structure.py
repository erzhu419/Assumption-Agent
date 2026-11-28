exec(open('/tmp/probe.py').read().split('m=old._panel_maps')[0])
from scipy.optimize import least_squares
m=load('/holdout_task/node_00.py'); z=m._latent(a,x); t=s['train']['traj']; y=s['train']['y']; n=52
u,v=b['u'],b['v']; print('panels stats',s['telemetry_vec'],'B range',u.min(0),u.max(0),'latent',z.min(0),z.max(0));
from itertools import combinations
for name,g in [('separate polynomial',lambda z:np.c_[z,z*z,z**3,z**4,z**5]),('pairs',lambda z:np.c_[z,z**3,*[z[:,i]*z[:,j] for i,j in combinations(range(4),2)]]),('sine',lambda z:np.c_[z,np.sin(2.15*z)]),('panels',lambda z:np.c_[m._panel_b(b,z),z]),('rational',lambda z:np.c_[z,z**3/(1+1.25*z*z)]),('all',lambda z:np.c_[z,z**2,z**3,np.sin(2.15*z),z**3/(1+1.25*z*z),*[z[:,i]*z[:,j] for i,j in combinations(range(4),2)]])]:
 d=lin(g(z)); pred=d[n:]@ls(d[:n],t[:n]); print('traj',name, np.mean(abs(pred-t[n:])),np.sqrt(np.mean((d@ls(d,t)-t)**2)))
for label,g,p in [('cube_rat',lambda u,h:u**3/(1+h[0]*u*u),[1.25]),('z_rat',lambda u,h:u/(1+h[0]*u*u),[1.25]),('offset_rat',lambda u,h:(u**3-h[1]*u)/(1+h[0]*u*u),[1.25,.1]),('linear_sine',lambda u,h:u-h[1]*np.sin(h[0]*u),[2.15,.3]),('linear_tanh',lambda u,h:u-h[1]*np.tanh(h[0]*u),[2.15,.3]),('rat_plus_square',lambda u,h:(u**3+h[1]*u*u+h[2]*u)/(1+h[0]*u*u),[1.25,0,.1]),('cube_plus_sine',lambda u,h:u**3-h[1]*np.sin(h[0]*u),[2.15,.3])]:
 def f(h,ret=False):
  d=lin(g(u,h)); c=ls(d,v); return c if ret else (d@c-v).ravel()
 h=least_squares(f,p,max_nfev=100).x;c=f(h,True);zz=lin(g(z,h))@c; d=lin(zz); print('B',label,h,np.sqrt(np.mean(f(h)**2)),np.mean(abs(d[n:]@ls(d[:n],t[:n])-t[n:])))
