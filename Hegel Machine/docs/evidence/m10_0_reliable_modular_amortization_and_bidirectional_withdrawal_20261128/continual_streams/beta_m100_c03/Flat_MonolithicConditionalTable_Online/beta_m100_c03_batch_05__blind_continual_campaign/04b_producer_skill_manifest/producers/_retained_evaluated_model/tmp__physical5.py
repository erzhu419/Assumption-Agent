import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import batch_solver as m
import numpy as np
from scipy.sparse import csc_matrix,eye,kron,diags
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; y=m._calibrate(s,s['y_obs_raw']); Z=s['Z_nuisance']; r=s['source_ridge']; nr=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T.copy(); ii=s['sensor_ij']@np.array([1,nx]);
def resp(p):
 axes=[]
 for j,(d,n) in enumerate(zip('xy',(nx,ny))):
  k,v,h=s['kappa_'+d],s['vel_'+d],s['L'+d]/(n-1); lo,hi=p[2*j:2*j+2]; a=m._axis(k,v,h,0).tolil()
  for i,neighbor,value in [(0,1,lo),(n-1,n-2,hi)]:
   if value=='P':
    left,right=(i-1)%n,(i+1)%n; km=(k[i]+k[left])*.5/h**2+max(v[i],0)/h; kp=(k[i]+k[right])*.5/h**2+max(-v[i],0)/h
    for pos,num in [(i,km+kp),(left,-km),(right,-kp)]:a[i,pos]=num
   else:a[i,i]+=value*s['robin_'+d]/h
  axes.append(a.tocsc())
 A=(kron(eye(ny),axes[0])+kron(axes[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc();return np.vstack([splu(A+mu*eye(nx*ny,format='csc')).solve(rhs)[ii] for mu in s['mu_shifts']])
def score(S,t):
 z=Z[t]; a=S[t]; b=y[t]; P=np.linalg.solve(z.T@z+nr*np.eye(z.shape[1]),z.T); A=a-z@(P@a); B=b-z@(P@b); w,_=m._box_qp(A.T@A+r*np.eye(A.shape[1]),-A.T@B,cap,budget); return y-S@w-Z@(P@(b-a@w))
t=np.ones(len(y),bool); group=np.tile(np.repeat(np.arange(8),4),3); results=[]
for p in itertools.product([0.,1.,'P'],repeat=4):
 S=resp(p); e=score(S,t); results.append((np.linalg.norm(e)/np.sqrt(len(e)),p))
print('best discrete', sorted(results,key=lambda a:a[0])[:8]);print('open residual',score(resp([1.]*4),t).reshape(3,-1).std(axis=1));
for grp in [group,np.repeat(np.arange(3),32)]:
 for flag in [False,True]:
  e=np.empty(len(y)); rates=[]
  for j in np.unique(grp):
   a=grp!=j; f=least_squares(lambda p:score(resp(p),a)[a],np.ones(4),bounds=(0.,4.),max_nfev=25).x if flag else np.ones(4); rates.append(f); e[~a]=score(resp(f),a)[~a]
  print('CV free',flag,'pooled/worst',np.sqrt(np.mean(e*e)),max(np.sqrt(np.mean(e[grp==j]**2)) for j in np.unique(grp)),'mean rates',np.mean(rates,axis=0))
