import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c01_batch_05_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; C=s['sensor_calibration_sweeps']; X,Y,B=[C[k] for k in ('ref_rosette','raw_rosette','single_bank_sweeps')]; good=np.max(abs(Y),axis=1)<5
for f in range(4):
 mask=np.arange(len(X))%4==f; bm=np.arange(len(B['bank_id']))%4!=f; p=old._fit_calibration(X[~mask],Y[~mask],{k:v[bm] for k,v in B.items()}); d=(old._response(X[mask&good],p)-Y[mask&good]); print('cal fold',f,'rms',np.mean(d*d,axis=0)**.5,'max',abs(d).max(),'params',p[8:])
y=old._calibrate(s); Z=s['Z_nuisance']; cap,tot=s['u_box'],s['total_budget']; ns=len(s['sensor_ij']); nshift=len(s['mu_shifts']); rosette=np.arange(len(y)).reshape(-1,4)[:,0]//4
print('shape',ns,nshift,Z.shape,s['source_modes'].shape,'shift',s['mu_shifts'],'robin',s['robin_x'],s['robin_y'],'calibrated range',y.min(),y.max()); print('indices',s['sensor_ij'])
def fit(A,z,b,ridge):
 P=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); D=A-z@(P@A); t=b-z@(P@b); w,_=old._box_simplex(D.T@D+ridge*np.eye(A.shape[1]),-D.T@t,cap,tot); return w,P@(b-A@w)
for walls in map(''.join,itertools.product('RFP',repeat=2)):
 A,U=old._transport(s,walls); line=[]
 for ridge in [1e-10,s['source_ridge']]:
  w,g=fit(A,Z,y,ridge); loss=np.sqrt(np.mean((A@w+Z@g-y)**2)); scores=[]
  for scheme in [np.repeat(rosette%4,4),np.arange(len(y))//ns]:
   v=[]
   for f in np.unique(scheme):
    t=scheme!=f; w,g=fit(A[t],Z[t],y[t],ridge); v.append(np.mean((A[~t]@w+Z[~t]@g-y[~t])**2)**.5)
   scores.append(np.round(v,5).tolist())
  line.append((round(loss,6),scores))
 print(walls,line)
