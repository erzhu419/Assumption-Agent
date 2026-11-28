from bridge_probe import D,X,Y,N
import numpy as np
from scipy.optimize import least_squares
S=D['calibration_load_cells']['single_channel_sweeps']; pair=[2,3,0,1]
clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<8
for j in range(4):
 m=S['lab_id']==j; print('sweep',j,np.round(np.c_[S['true_stress'][m],S['cmd_stress'][m]],6).tolist()); p=pair[j]; m=(np.count_nonzero(X,axis=1)==1)&(X[:,p]!=0)&clean;print('cross axis',j,np.round(np.c_[X[m,p],Y[m,j]],6).tolist())
forms={'sin':np.sin,'tanh':np.tanh,'rat':lambda q:q/(1+q*q),'atan':np.arctan,'zero':lambda q:q*0}

def fit(j,m,f,g,mode,sw=False):
 p=pair[j]; x=X[m]; a=x[:,j]; b=x[:,p]; t=Y[m,j]
 if sw:
  sm=(S['lab_id']==j)&(abs(S['cmd_stress']-np.median(S['cmd_stress']))<8); ss=S['true_stress'][sm]; a=np.r_[a,ss]; b=np.r_[b,np.zeros_like(ss)]; t=np.r_[t,S['cmd_stress'][sm]]
 def design(u,v,z):
  k,l,h=z; q=u if mode not in ('mixed','both') else u+h*v
  cols=[np.ones_like(u),u,v,v*v,forms[f](k*q),forms[g](l*v)]
  if mode in ('gate','both'): cols.extend([u*v, v*np.cos(k*q),v*v*np.cos(k*q)])
  if mode=='diag2': cols.extend([u*u,u**3,np.sin(l*u)])
  return np.array(cols).T
 def res(z):
  A=design(a,b,z); return A@np.linalg.lstsq(A,t,rcond=1e-10)[0]-t
 p0={'sin':2.5,'tanh':1.5,'rat':.9,'atan':2.,'zero':1.}; bounds=([.2,.2,-.8],[5,6,.8]); sol=min((least_squares(res,[p0[f],l,h],bounds=bounds,max_nfev=100,ftol=1e-10) for l,h in [(p0[g],0.),(2.8,-.15)]),key=lambda r:r.cost); z=sol.x; C=np.linalg.lstsq(design(a,b,z),t,rcond=1e-10)[0]
 return design(X[:,j],X[:,pair[j]],z)@C,z,C

specs=[(f,g,mode) for f in ['sin','tanh','rat'] for g in ['zero','sin','tanh','rat'] for mode in ['add','mixed','gate','both']]+[(f,'zero','diag2') for f in ['sin','tanh','rat']]
folds=np.random.default_rng(8).permutation(N)%5; edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
for sw in [False,True]:
 rows=[]
 for f,g,mode in specs:
  r=[fit(j,clean,f,g,mode,sw) for j in range(4)]; e=np.column_stack([v[0] for v in r])-Y; rows.append((np.sqrt(np.mean(e[clean]**2)),f,g,mode,r))
 for rms,f,g,mode,r in sorted(rows,key=lambda x:x[0])[:6]:
  pred=np.zeros_like(Y)
  for fold in range(5):
   h=folds==fold
   for j in range(4): pred[h,j]=fit(j,clean&~h,f,g,mode,sw)[0][h]
  ep=np.column_stack([fit(j,clean&~edge,f,g,mode,sw)[0] for j in range(4)])
  print(sw,f,g,mode,'rms/oof p90/max/edge max',rms,*np.quantile(np.linalg.norm((pred-Y)[clean],axis=1),[.9,1]),max(np.linalg.norm((ep-Y)[edge&clean],axis=1)), 'z',np.round([v[1] for v in r],4).tolist())
 best=min(rows,key=lambda v:v[0]); print('best coeff',np.round(np.array([v[2] for v in best[-1]]),6).tolist())
