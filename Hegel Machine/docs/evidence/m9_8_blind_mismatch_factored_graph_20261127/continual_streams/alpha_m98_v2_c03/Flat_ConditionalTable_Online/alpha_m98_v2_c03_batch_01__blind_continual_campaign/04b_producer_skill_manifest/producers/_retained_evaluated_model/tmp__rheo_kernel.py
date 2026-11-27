exec(open('/tmp/rheo_probe.py').read().split("print('train RMSE'")[0])
from itertools import combinations
P,H,D=c; n=len(H); q=len(t)+1
B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]])
A=np.linalg.lstsq(B,yd.T,rcond=None)[0].T
Z=np.c_[np.ones(n),H,P]; C=np.linalg.lstsq(Z,A,rcond=None)[0]; print('linear amplitude coefficients by 1,hold,past0..5\n',C.round(4))
print('linear amp residual rms',np.std(A-Z@C,0)); print('shared-tau noise level',np.std(yd-A@B.T,axis=0).round(4)); print('baseline normalized memory impulse',np.array([(-np.expm1(-dt[i]/tau))*np.exp(-sum(dt[i+1:])/tau) for i in range(len(dt))]).round(4));
print('aggregate coefficients',C.sum(0));print('branch ratios',C[:,:2]/w[:2]); print('sweep fit',[(int(g),np.polyfit(d['calibration_load_cells']['single_channel_sweeps']['true_stress'][ss],d['calibration_load_cells']['single_channel_sweeps']['cmd_stress'][ss],1).round(4)) for g in np.unique(d['calibration_load_cells']['single_channel_sweeps']['lab_id']) for ss in [d['calibration_load_cells']['single_channel_sweeps']['lab_id']==g]])
for name,Z0 in [('affine_stress',Z),('affine+features',np.c_[Z,D[:,1:]]),('affine+hold2',np.c_[Z,H**2]),('affine+last2',np.c_[Z,P[:,-1]**2]),('affine+labs',np.c_[Z,np.eye(4)[dg['lab_id']][:,1:]])]:
 for nb in [2,3]:
  b0=B[:,:nb]; az=np.einsum('ij,kl->ikjl',Z0,b0).reshape(n*q,-1); yh=yd.ravel(); pred=np.empty_like(yd)
  for g in np.unique(dg['lab_id']):
   mask=dg['lab_id']==g; ix=np.repeat(~mask,q); p0=np.linalg.lstsq(az[ix],yh[ix],rcond=None)[0]; pred[mask]=(az[~ix]@p0).reshape(-1,q)
  print(name,nb,'LOLO rms',[(round(np.sqrt(np.mean((pred[ss,:-1]-yd[ss,:-1])**2)),4),round(np.sqrt(np.mean((pred[ss,-1]-yd[ss,-1])**2)),4)) for g in np.unique(dg['lab_id']) for ss in [dg['lab_id']==g]])
print('error correlations with steps, hold, max, min, mean, last jump, features, lab'); E=A[:,:2]-(Z@C)[:,:2]; extras=np.c_[P,H,P.max(1),P.min(1),P.mean(1),H-P[:,-1],D[:,1:],dg['lab_id']]; print(np.corrcoef(np.c_[E,extras].T)[:2,2:].round(2));
