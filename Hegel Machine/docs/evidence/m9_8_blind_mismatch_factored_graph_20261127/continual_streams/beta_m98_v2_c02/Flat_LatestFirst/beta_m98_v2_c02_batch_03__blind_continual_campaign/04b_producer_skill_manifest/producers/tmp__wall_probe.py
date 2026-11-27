import pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; d=s['sensor_calibration_sweeps']; r=d['ref_rosette']; raw=d['raw_rosette']; X=np.c_[np.ones(len(r)),r]; B=np.linalg.lstsq(X,raw,rcond=None)[0]
print('cal RMSE',np.sqrt(np.mean((X@B-raw)**2,axis=0)),'coef',B.tolist())
nx,ny=s['nx'],s['ny']; ns=len(s['sensor_ij']); nk=len(s['mu_shifts']); pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]); obs=np.array([np.linalg.solve(B[1:].T,(p['y_pulse_raw'].reshape(-1,4)-B[0]).T).T.ravel() for p in pulses]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; print('shifts',s['mu_shifts'],'boundary sensors',np.sum((s['sensor_ij']==0)|(s['sensor_ij']==[nx-1,ny-1]),axis=0));
def axis(axis,p):
 k=s['kappa_'+axis]; v=s['vel_'+axis]; h=s['L'+axis]/(len(k)-1); rob=s['robin_'+axis]; face=(k[1:]+k[:-1])/(2*h*h); lo=-face-np.maximum(v[1:],0)/h; up=-face-np.maximum(-v[:-1],0)/h; lo[-1]=-k[-1]/h**2*p[3]; up[0]=-k[0]/h**2*p[2]; dg=np.r_[-up[0]+rob/h*p[0],-lo[:-1]-up[1:],-lo[-1]+rob/h*p[1]]; a=sparse.diags([lo,dg,up],[-1,0,1],format='lil'); a[0,-1]=-p[4]*k[0]/h**2; a[-1,0]=-p[5]*k[-1]/h**2; a[0,0]-=a[0,-1]; a[-1,-1]-=a[-1,0]; return a.tocsc()
src=s['source_modes'].reshape(-1,nx*ny,order='F').T@W.T; ix,iy=s['sensor_ij'].T; rows=ix+nx*iy; ident=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(p):
 L=sparse.kron(sparse.eye(ny),axis('x',p[:6]),format='csc')+sparse.kron(axis('y',p[6:]),sparse.eye(nx),format='csc')+V; return np.hstack([splu(L+mu*ident).solve(src)[rows,:].T for mu in s['mu_shifts']])
z=np.tile([0.,0,1,1,0,0],2); open_=np.tile([1.,1,1,1,0,0],2)
rmse=lambda r: np.sqrt(np.mean(r*r,axis=-1))
for name,p in [('sealed',z),('open',open_)]:
 e=pred(p)-obs; print(name,'by pulse/shift',rmse(e.reshape(-1,nk,ns)),'projected',rmse(e-e@Q@Q.T))
groups={'shared leak':[[0,1,6,7]],'axis leak':[[0,1],[6,7]],'four leaks':[[0],[1],[6],[7]],'four leak and coupling':[[j] for j in [0,1,6,7,2,3,8,9]],'plus wrap':[[j] for j in range(12)]}
for project in [False,True]:
 for name,groups_ in groups.items():
  D=np.zeros((12,len(groups_)))
  for i,g in enumerate(groups_): D[g,i]=1
  def res(a):
   e=pred(z+D@a)-obs; return e-e@Q@Q.T if project else e
  for tr in ([0],[1],[0,1]):
   opt=least_squares(lambda a:res(a)[tr].ravel(),np.zeros(D.shape[1]),bounds=(-.8,3),max_nfev=60); e=res(opt.x); print('project',project,name,'fit',tr,'coef',np.round(opt.x,5),'slice',np.round(rmse(e.reshape(-1,nk,ns)),7).tolist())
