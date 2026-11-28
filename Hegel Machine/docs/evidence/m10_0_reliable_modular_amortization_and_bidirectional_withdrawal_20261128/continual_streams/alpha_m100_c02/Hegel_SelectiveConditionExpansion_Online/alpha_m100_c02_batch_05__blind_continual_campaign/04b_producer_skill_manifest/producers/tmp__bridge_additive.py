from bridge_probe import D,X,Y,N
import numpy as np
from scipy.optimize import least_squares
P=np.array([2,3,0,1]); S=D['calibration_load_cells']['single_channel_sweeps']; SX=np.eye(4)[S['lab_id']]*S['true_stress'][:,None]; SY=S['cmd_stress']; clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<8; sc=abs(SY-np.median(SY))<8
forms=[np.sin,np.tanh,np.arctan,lambda q:q/(1+q*q),lambda q:q/np.sqrt(1+q*q)]
def fit(j,m,typ,deg,interaction=False,sm=sc):
 p=P[j]; x=np.r_[X[m],SX[sm&(S['lab_id']==j)]]; y=np.r_[Y[m,j],SY[sm&(S['lab_id']==j)]]
 def design(x,z):
  u=x[:,j]; v=x[:,p]; cols=[u*0+1,u,v,v*v];
  if typ=='poly': cols.extend(u**i for i in range(3,deg+1,2))
  else:
   cols.extend(forms[k](z[a]*u) for a,k in enumerate(typ));
   if deg: cols.append(u**3)
  cols.extend((v**3,v**4,v**5))
  if interaction: cols.extend((u*v,u*u*v,u*v*v))
  return np.array(cols).T
 def obj(z):
  a=design(x,z); return a@np.linalg.lstsq(a,y,rcond=1e-9)[0]-y
 if typ=='poly': z=[]
 else:
  z=min((least_squares(obj,q,bounds=(.25,10.),max_nfev=160,gtol=1e-11) for q in ([[2.5],[1.1]] if len(typ)==1 else [[2.5,1.3],[2.5,4.6],[1.1,2.5]])),key=lambda r:r.cost).x
 b=np.linalg.lstsq(design(x,z),y,rcond=1e-9)[0]; return design(X,z)@b, design(SX,z)@b,z,b
spec=[('poly',i,g) for i in [5,7,9,11] for g in [False,True]]+[(t,d,g) for t in [(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(1,1),(1,3),(3,3)] for d,g in [(0,False),(1,False),(0,True)]]
edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8); folds=np.random.default_rng(8).permutation(N)%5
rows=[]
for s in spec:
 r=[fit(j,clean,*s) for j in range(4)]; ef=np.linalg.norm((np.array([v[0] for v in r]).T-Y),axis=1); rows.append((np.sqrt(np.mean(ef[clean]**2)),s,r))
for rms,s,r in sorted(rows,key=lambda r:r[0])[:12]:
 pred=np.empty_like(Y)
 for f in range(5):
  h=folds==f
  for j in range(4): pred[h,j]=fit(j,clean&~h,*s)[0][h]
 e=np.linalg.norm(pred-Y,axis=1); ep=np.array([fit(j,clean&~edge,*s)[0] for j in range(4)]).T
 print(s, 'rms/p90/max/edge',*[round(v,7) for v in (rms,np.quantile(e[clean],.9),max(e[clean]),max(np.linalg.norm((ep-Y)[edge&clean],axis=1)))], 'z', [np.round(v[2],4).tolist() for v in r]);
 if rms<.002: print('coeffs',np.round([v[3] for v in r],5))
# Test whether subtracting the independently interpolated diagonal leaves a univariate paired curve.
from scipy.interpolate import CubicSpline
for j in range(4):
 sm=sc&(S['lab_id']==j); f=CubicSpline(S['true_stress'][sm],SY[sm]); v=X[clean,P[j]]; u=X[clean,j]; e=Y[clean,j]-f(u); A=np.c_[v,v*v,v**3,v**4,v**5]; b=np.linalg.lstsq(A,e,rcond=None)[0]; print('paired decomposition',j,np.round(b,5),np.quantile(abs(e-A@b),[.5,.9,1]))
