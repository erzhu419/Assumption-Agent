import pickle,numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=160)
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; s=c['single_channel_sweeps']; L=c['true_channel_loads']; R=c['bridge_raw']; good=np.max(abs(R),axis=1)<4; L,R=L[good],R[good]; folds=np.arange(len(L))%5
for j in range(4):
 k=(j+2)%4; u,v=L[:,j],L[:,k]; y=R[:,j]; sel=(s['lab_id']==j)&(abs(s['cmd_stress'])<4); x=s['true_stress'][sel]; t=s['cmd_stress'][sel]; f=CubicSpline(x,t); d=y-f(u); print('\nchannel',j,'diagonal samples',np.c_[x,t].T)
 print('opposite slopes sample',np.c_[u,v,d,d/np.where(abs(v)>.02,v,np.nan)][-14:])
 for label,fun,p0 in [('shift',lambda p,u,v:f(u+p[0]*v),[-.15]),('shift plus quadratic',lambda p,u,v:f(u+p[0]*v+p[1]*v*v+p[2]*u*v)+p[3]*v+p[4]*v*v,[-.15,.04,0,0,0]),('additive interaction',lambda p,u,v:f(u)+p[0]*v+p[1]*v*v+p[2]*v**3+p[3]*v*u+p[4]*v*u*u+p[5]*v*v*u+p[6]*v*u**3,[-.2,.03,0,0,0,0,0])]:
  e=[]
  for z in range(5):
   m=folds!=z; p=least_squares(lambda p:fun(p,u[m],v[m])-y[m],p0).x; e.extend(fun(p,u[~m],v[~m])-y[~m]);
  p=least_squares(lambda p:fun(p,u,v)-y,p0).x; print(label,p,'cv rmse',np.sqrt(np.mean(np.square(e))))
 for deg in [3,5,7,9]:
  A=np.array([x**i for i in [0,*range(1,deg+1,2)]]).T; p=np.linalg.lstsq(A,t,rcond=None)[0]; print('odd degree',deg,'rmse',np.sqrt(np.mean((A@p-t)**2)),p)
