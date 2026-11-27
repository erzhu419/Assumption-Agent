import sys,json,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from scipy.sparse import diags,kron,eye,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
root='/holdout_task/'
a=pickle.load(open(root+'public_batch.pkl','rb'))[0]; pay=json.load(open(root+'_upstream_param_payload.json')); st=calibrate_upstream(a['study'],pay)
pulses=a['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in pulses],axis=1)
Y=np.stack([calibrate_upstream(dict(a['study'],y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses],axis=1)
z=st['Z_nuisance']; proj=np.linalg.lstsq(z,np.eye(len(z)),rcond=None)[0]
nx,ny=st['nx'],st['ny']; ix,iy=st['sensor_ij'].T; N=nx*ny
M=st['source_modes']; F=np.asfortranarray(M.transpose(2,1,0).reshape(N,len(M)))
B=F@W
I=eye(N,format='csc'); V=diags(st['V_reaction_2d'].ravel(order='F'),format='csc')

def axis(k,v,h,r,p):
 d0,d1,c0,c1,b0,b1=p
 kp=.5*(k[1:]+k[:-1])/h**2
 lo=-kp-np.maximum(v[1:],0)/h; up=-kp-np.maximum(-v[:-1],0)/h
 lo[-1]=-c1*k[-1]/h**2; up[0]=-c0*k[0]/h**2
 diag=np.r_[c0*k[0]/h**2+b0*k[0]/h**2+d0*r/h,-lo[:-1]-up[1:],c1*k[-1]/h**2+b1*k[-1]/h**2+d1*r/h]
 A=diags([lo,diag,up],[-1,0,1],format='lil'); A[0,-1]=-b0*k[0]/h**2; A[-1,0]=-b1*k[-1]/h**2
 return A.tocsc()

def forward(p,rhs=B):
 Ax,Ay=[axis(st['kappa_'+q],st['vel_'+q],st['L'+q]/(n-1),st['robin_'+q],p0) for q,n,p0 in zip('xy',(nx,ny),np.reshape(p,(2,6)))]; L=(kron(eye(ny),Ax)+kron(Ay,eye(nx))+V).tocsc()
 U=[splu(L+mu*I).solve(np.asfortranarray(rhs)) for mu in st['mu_shifts']]
 return np.concatenate([u[ix+nx*iy] for u in U]),U[-1]

def err(p, nuisance=False):
 r=forward(p)[0]-Y
 return r-z@(proj@r) if nuisance else r

def metrics(e):
 return np.round(np.sqrt(np.mean(e**2,axis=0)),9)

if __name__=='__main__':
 for name,p in [('Robin',[1,1,1,1,0,0]),('reflecting',[0,0,1,1,0,0]),('ghost mirror',[0,0,2,2,0,0]),('periodic',[0,0,1,1,1,1]),('one sided Dirichlet',[1,1,0,0,0,0])]:
  print(name,metrics(err(p*2)),metrics(err(p*2,True)))
 p0=np.array([1,1,1,1,0,0]*2,dtype=float)
 for free in ([0,1,6,7],[0,1,2,3,6,7,8,9],list(range(12))):
  for nuisance in (False,True):
   def unpack(x):
    p=p0.copy(); p[free]=x; return p
   lo=np.array([0,0,0,0,0,0]*2); hi=np.array([100,100,15,15,15,15]*2)
   for held in (-1,0,1,2):
    keep=np.arange(3)!=held
    fit=least_squares(lambda x:err(unpack(x),nuisance)[:,keep].ravel(),p0[free],bounds=(lo[free],hi[free]),max_nfev=100,gtol=1e-12,xtol=1e-12)
    p=unpack(fit.x); print('fit',len(free),'nuis',nuisance,'held',held,'p',p.round(5),'rmse',metrics(err(p,nuisance)))
 print('Y ranges',Y.min(0),Y.max(0))
 sweep=st['sensor_calibration_sweeps']; raw=sweep['raw_rosette']; ref=sweep['ref_rosette']; good=(abs(raw)<=8.5).all(axis=1); print('canonical sweep delta',np.max(abs(raw[good]-ref[good])), 'outlier rows',sum(~good))
