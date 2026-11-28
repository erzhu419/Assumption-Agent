import sys,json,pickle,itertools
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
W=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]).T
Y=np.array([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in item['campaign_diagnostics']]).T
nx,ny=s['nx'],s['ny']; dims=(nx,ny); N=nx*ny
rhs=s['source_modes'].transpose(0,2,1).reshape(len(W),N).T@W
ij=s['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; eyeN=eye(N,format='csc')
V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
def axis(key,args):
 k=s['kappa_'+key]; v=s['vel_'+key]; b=s['robin_'+key]; h=s['L'+key]/(len(k)-1)
 f=(k[:-1]+k[1:])/(2*h*h); lo=np.r_[-f[:-1]-np.maximum(v[1:-1],0)/h,0.]; hi=np.r_[0.,-f[1:]-np.maximum(-v[1:-1],0)/h]; main=np.r_[0.,-lo[:-1]-hi[1:],0.]; A=diags([lo,main,hi],[-1,0,1],format='lil')
 for i,j,face,vd,sg,(m,r,c,a,d) in zip((0,len(k)-1),(1,len(k)-2),(f[0],f[-1]),(v[0],v[-1]),(-1,1),args):
  diff=m*k[i]/h**2+c*face+a*max(sg*vd,0)/h; A[i,j]=-diff; A[i,i]=diff+r*b/h+d*max(-sg*vd,0)/h
 return A.tocsc()
def predict(ax,ay):
 L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.concatenate([splu(L+mu*eyeN).solve(rhs)[rows] for mu in s['mu_shifts']])
def errs(pred):
 E=pred-Y; F=E-Q@(Q.T@E)
 return np.sqrt(np.mean(E**2,axis=0)),np.sqrt(np.mean(F**2,axis=0))
def named(m=1,r=1,c=0,a=0,d=0): return (m,r,c,a,d)
if __name__=='__main__':
 print('pulse weights',W.T,'response RMS',np.sqrt(np.mean(Y**2,axis=0)))
 types={'R':named(), 'N':named(r=0),'R2':named(2,2),'N2':named(2,0),'Rdiff2':named(2,1),'Rface':named(0,1,1),'Nface':named(0,0,1),'Radv':named(a=1,d=1),'Nadv':named(r=0,a=1,d=1),'Rhalf':named(1,.5),'D':named(0,10)}
 cache={key:{(p,q):axis(key,(types[p],types[q])) for p,q in itertools.product(types,repeat=2)} for key in 'xy'}
 E=[]
 for pairx,ax in cache['x'].items():
  for pairy,ay in cache['y'].items():
   e,f=errs(predict(ax,ay)); E.append((pairx+pairy,e,f))
 for n in (0,1,2,3):
  idx=[i for i in range(3) if i!=n]
  print('omit',n if n<3 else 'none',[(g,np.round(e,7).tolist(),np.round(f,7).tolist()) for g,e,f in sorted(E,key=lambda z:np.mean(z[1][idx]**2))[:4]])
 for g,e,f in E:
  if g in [('R',)*4,('N',)*4]:print('baseline',g,e,f)
