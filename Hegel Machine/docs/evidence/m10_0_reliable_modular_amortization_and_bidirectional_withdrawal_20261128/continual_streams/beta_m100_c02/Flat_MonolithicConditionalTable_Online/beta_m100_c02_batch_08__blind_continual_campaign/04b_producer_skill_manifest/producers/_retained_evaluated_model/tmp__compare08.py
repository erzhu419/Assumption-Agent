import itertools, numpy as np
import bound08 as b
s,m=b.s,b.m
Z=s['Z_nuisance']; f=b.fit(b.X,b.Y,'sep'); y=b.calibrate(f)
print('archive calibration comparison'); x=b.X; yy=b.Y; c=m.calibration_map(s['sensor_calibration_sweeps']); print('archived fit RMSE',np.sqrt(np.mean((m.raw_map(x,c)-yy)**2,axis=0)))
# Keep all three shifts in spatial folds; also test extrapolation to a whole shift.
groups=[np.arange(len(y))//32]+[np.tile(np.arange(32)//4,3)]
for i,j in itertools.product([1,2],repeat=2):
 S,U=m.chamber_response(s,[[0,np.nan][i-1]]*2+[[0,np.nan][j-1]]*2)
 errs=[]
 for g in groups:
  es=[]
  for k in np.unique(g):
   train=g!=k; A=S[train]; z=Z[train]; t=y[train]; q=np.linalg.qr(z,mode='reduced')[0]; a=A-q@(q.T@A); t0=t-q@(q.T@t); w,_=m.box_simplex(a.T@a+1e-10*np.eye(a.shape[1]),a.T@t0,s['u_box'],s['total_budget']); gamma=np.linalg.lstsq(z,t-A@w,rcond=None)[0]; es.append(np.sqrt(np.mean((S[~train]@w+Z[~train]@gamma-y[~train])**2)))
  errs.append(np.array(es))
 print('walls',i,j,'by shift',errs[0].round(6),'spatial RMS/worst',np.sqrt(np.mean(errs[1]**2)),max(errs[1]))
for train in [np.arange(len(x))%5!=k for k in range(5)]:
 d=s['sensor_calibration_sweeps'].copy(); d['ref_rosette']=x[train]; d['raw_rosette']=yy[train]; c=m.calibration_map(d); e=m.raw_map(x[~train],c)-yy[~train]; print('archived cal CV',np.sqrt(np.mean(e**2,axis=0)))
