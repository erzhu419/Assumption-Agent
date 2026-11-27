import pickle, runpy
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
old=runpy.run_path('/tmp/hegel_continual_archive/beta_m98_v2_c01_s00_solver.py')
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; p=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]
W=np.column_stack([a['w_pulse'] for a in p]); Y=np.column_stack([old['_calibrate'](dict(s,y_obs_raw=a['y_pulse_raw'])) for a in p]).reshape(3,len(rows),-1)
R=np.stack([x.ravel(order='F') for x in s['source_modes']],axis=1)@W
base=[]; edge=[]; face=[]; robin=[]
for ax,n in [('x',nx),('y',ny)]:
 h=s['L'+ax]/(n-1); k=s['kappa_'+ax]; rob=s['robin_'+ax]; base.append(old['_axis'](k,s['vel_'+ax],h,rob).toarray()); edge.extend(k[[0,-1]]/h**2); face.extend(0.5*(k[[0,-1]]+k[[1,-2]])/h**2); robin.extend([rob/h]*2)
edge,face,robin=map(np.array,(edge,face,robin)); ident=eye(nx*ny,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def predict(a,b,c):
 axes=[x.copy() for x in base]
 for k,axis in enumerate(axes):
  for j,(i,inn,opp) in enumerate([(0,1,-1),(-1,-2,0)],start=2*k):
   axis[i,i]=a[j]+b[j]+c[j]; axis[i,inn]=-a[j]; axis[i,opp]=-c[j]
 L=kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')+reaction
 return np.array([splu(L+mu*ident).solve(R)[rows] for mu in s['mu_shifts']])
for name,a,b,c in [('robin',edge,robin,edge*0),('reflecting kappa',edge,edge*0,edge*0),('reflecting face',face,edge*0,edge*0),('doubled reflecting kappa',2*edge,edge*0,edge*0),('doubled reflecting face',2*face,edge*0,edge*0),('periodic kappa',edge,edge*0,edge),('periodic face',face,edge*0,np.repeat([edge[:2].mean(),edge[2:].mean()],2))]:
 d=predict(a,b,c)-Y; print(name, np.sqrt(np.mean(d*d,axis=1)).round(7).tolist())
for free in (8,12):
 for train in [np.ones(Y.shape,dtype=bool),np.indices(Y.shape)[2]==0,np.indices(Y.shape)[2]==1]:
  def fun(x): return predict(x[:4]*edge,x[4:8]*robin,x[8:]*edge if free==12 else edge*0)-Y
  x0=np.r_[np.ones(4),np.full(4,0.05),np.full(free-8,0.05)]; lo=np.r_[np.full(4,0.01),np.full(4,-0.18),np.full(free-8,-0.05)]; hi=np.full(free,8.)
  opt=least_squares(lambda x: fun(x)[train],x0,bounds=(lo,hi),gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=180)
  print('fit',free,train.sum(),opt.x.round(6),'rms by shift/pulse',np.sqrt(np.mean(fun(opt.x)**2,axis=1)).round(8).tolist())
print('edge/face/robin',edge,face,robin)
