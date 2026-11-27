import pickle, numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; pulses=it['campaign_diagnostics']
nx,ny=s['nx'],s['ny']; W=np.stack([p['w_pulse'] for p in pulses],axis=1)
ref=s['sensor_calibration_sweeps']['ref_rosette']; raw=s['sensor_calibration_sweeps']['raw_rosette']; C=np.linalg.lstsq(np.c_[ref,np.ones(len(ref))],raw,rcond=None)[0]
Y=np.stack([np.linalg.solve(C[:-1].T,(p['y_pulse_raw'].reshape(-1,4)-C[-1]).T).T.ravel() for p in pulses],axis=1)
R=np.stack([u.ravel(order='F') for u in s['source_modes']],axis=1)@W
ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]; I=eye(nx*ny,format='csc'); rx,ry=s['robin_x'],s['robin_y']
def axis(a,p):
 n=s['n'+a]; h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; f=(k[:-1]+k[1:])/(2*h*h)
 lo=-f-np.maximum(v[1:],0)/h; hi=-f-np.maximum(-v[:-1],0)/h
 l,r,sl,sr,wl,wr=p; hi[0]=-sl*k[0]/h**2; lo[-1]=-sr*k[-1]/h**2
 d=-np.r_[0.,lo]-np.r_[hi,0.]; d[0]+=l/h+wl*k[0]/h**2; d[-1]+=r/h+wr*k[-1]/h**2
 A=diags((lo,d,hi),(-1,0,1),format='lil'); A[0,-1]=-wl*k[0]/h**2; A[-1,0]=-wr*k[-1]/h**2
 return A.tocsc()
def error(p):
 A=kron(eye(ny),axis('x',p[:6]),format='csc')+kron(axis('y',p[6:]),eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.vstack([splu(A+mu*I).solve(R)[idx] for mu in s['mu_shifts']])-Y
def scores(e):
 return np.sqrt(np.mean(e.reshape(3,-1,3)**2,axis=1)).round(8).tolist()
for name,p in [('Robin',[rx,rx,1,1,0,0,ry,ry,1,1,0,0]),('reflecting',[0,0,1,1,0,0]*2),('doubled face',[0,0,2,2,0,0]*2),('periodic endpoint',[0,0,1,1,1,1]*2),('x periodic',[0,0,1,1,1,1,0,0,1,1,0,0]),('y periodic',[0,0,1,1,0,0,0,0,1,1,1,1]),('x Robin y reflecting',[rx,rx,1,1,0,0,0,0,1,1,0,0]),('x reflecting y Robin',[0,0,1,1,0,0,ry,ry,1,1,0,0])]:
 print(name,scores(error(p)))
base=np.array([0.,0.,1.,1.,0.,0.]*2)
for cols in ([0,1,6,7],[0,1,2,3,6,7,8,9],list(range(12))):
 def expand(q):
  p=base.copy(); p[cols]=q; return p
 for omitted in [-1,0,1,2]:
  keep=[i for i in range(3) if i!=omitted]
  fit=least_squares(lambda q:error(expand(q))[:,keep].ravel(),base[cols],bounds=(np.array([0,0,.05,.05,0,0]*2)[cols],np.array([4,4,4,4,4,4]*2)[cols]),max_nfev=120,gtol=1e-11)
  print('fit',len(cols),'omit',omitted,'p',expand(fit.x).round(6).tolist(),'rmse',float(np.sqrt(np.mean(error(expand(fit.x))[:,omitted if omitted>=0 else keep]**2))))
