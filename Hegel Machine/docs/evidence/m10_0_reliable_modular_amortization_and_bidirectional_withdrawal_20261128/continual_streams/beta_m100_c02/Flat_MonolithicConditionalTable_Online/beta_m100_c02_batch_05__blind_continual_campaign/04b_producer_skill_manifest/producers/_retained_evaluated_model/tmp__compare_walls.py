import sys,json,pickle,itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c02_batch_04_solver as prev
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=prev.calibrate(s); Z=s['Z_nuisance']; n=s['nx']; m=s['ny']; st=s['sensor_ij']; ns=len(st); a=s['sensor_calibration_sweeps']; c=prev.calibration_map(a); print('calibration coefficient rows:',c); x=a['ref_rosette']; nxt=np.roll(x,-1,axis=1); e=a['raw_rosette']-(c[0]+c[1]*x+c[2]*x**3+c[3]*np.roll(x,1,axis=1)+c[4]*nxt+c[5]*x*nxt); print('cal residual abs quantiles',np.quantile(abs(e),[.5,.9,.95,.98,1]), 'y calibrated range',y.min(),y.max()); print('unique sensors and edges',len(np.unique(st,axis=0)),st[(st[:,0]==0)|(st[:,0]==n-1)|(st[:,1]==0)|(st[:,1]==m-1)])
def axis(key,spec):
 h=s['L'+key]/(s['n'+key]-1); k=s['kappa_'+key]; v=s['vel_'+key]; A=prev.axis_operator(k,v,h,periodic=spec=='p');
 if spec!='p':
  r=s['robin_'+key]/h; A=A+sparse.diags([spec[0]*r]+[0]*(len(k)-2)+[spec[1]*r],format='csc')
 return A
I=sparse.eye(n*m,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=s['source_modes'].reshape(-1,n*m,order='F').T; ids=st[:,0]+n*st[:,1]
def forward(ax,ay):
 L=sparse.kron(sparse.eye(m),ax,format='csc')+sparse.kron(ay,sparse.eye(n),format='csc')+V
 U=[splu(L+mu*I).solve(rhs) for mu in s['mu_shifts']]; return np.concatenate([u[ids] for u in U]),U[-1]
def fit(S,tr,source=.04):
 z=Z[tr]; sy=np.column_stack((S[tr],y[tr])); r=sy-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@sy); A,b=r[:,:-1],r[:,-1]; w,nu=prev.box_simplex(A.T@A+source*np.eye(A.shape[1]),A.T@b,s['u_box'],s['total_budget']); gamma=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@(y[tr]-S[tr]@w)); return w,y-S@w-Z@gamma
if __name__=='__main__':
 opts=['p',(0,0),(1,1),(1,0),(0,1)]; records=[]; tr=np.ones(len(y),bool)
 folds=[np.tile(np.arange(ns)//4==i,len(s['mu_shifts'])) for i in range(ns//4)]
 for p,q in itertools.product(opts,repeat=2):
  S,U=forward(axis('x',p),axis('y',q)); w,e=fit(S,tr); cv=np.array([np.mean(fit(S,~t)[1][t]**2)**.5 for t in folds]); _,e0=fit(S,tr,0.); records.append((np.mean(cv**2)**.5,p,q,np.mean(e**2)**.5,cv.max(),np.mean(e0**2)**.5))
 for x in sorted(records,key=lambda x:x[0]): print(x)
