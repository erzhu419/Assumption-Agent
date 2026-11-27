from transport_probe import *
A=op(0,2,1,1); h=np.array([s['Lx']/(nx-1),s['Ly']/(ny-1)]); r=np.array([s['robin_x'],s['robin_y']]); a0=np.repeat(r/h,2)
G=[np.array([[1,0],[1,0],[0,1],[0,1]]),np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1]]),np.eye(4)]
def solve(g,mask):
 a=np.linalg.lstsq(g,a0,rcond=None)[0]; return g@least_squares(lambda a:(predict(A,g@a)-Y)[mask].ravel(),a,gtol=1e-13,ftol=1e-12).x
masks=[np.tile((np.arange(len(idx))//4)%4==f,len(mu)) for f in range(4)]
masks += [np.arange(len(Y))//len(idx)==f for f in range(len(mu))]
for name,g in [('nominal',None),('axis',G[0]),('x shared',G[1]),('split',G[2])]:
 a=a0 if g is None else solve(g,np.ones(Y.shape,bool)); E=predict(A,a)-Y
 cv=[np.sqrt(np.mean((predict(A,a0 if g is None else solve(g,~mask))-Y)[mask]**2)) for mask in masks];
 pcv=[]
 for i in range(Y.shape[1]):
  train=np.ones_like(Y,dtype=bool); train[:,i]=False
  pcv.append(np.sqrt(np.mean((predict(A,a0 if g is None else solve(g,train))-Y)[:,i]**2)))
 print(name,'walls',a,'rmse',np.sqrt(np.mean(E**2)),'max',abs(E).max(),'rosette/shift cv',np.round(cv,8),'pulse cv',np.round(pcv,8))
 if g is G[-1]:
  print('residual by shift mean/std',np.round(E.reshape(len(mu),len(idx),-1).mean(1),8),np.round(E.reshape(len(mu),len(idx),-1).std(1),8)); print('residual Z slopes',np.linalg.lstsq(s['Z_nuisance'],E,rcond=None)[0]); print('residual wall rmse',[(i,np.sqrt(np.mean(E[np.tile(W[idx,i].astype(bool),len(mu))]**2))) for i in (0,2)])
print('rhs/cal noise',[(p['w_pulse'].sum(),) for p in pulses],np.std(s['sensor_calibration_sweeps']['raw_rosette']-s['sensor_calibration_sweeps']['ref_rosette']))
