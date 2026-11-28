import pickle,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sweeps=s['sensor_calibration_sweeps']; X=sweeps['ref_rosette']; Y=sweeps['raw_rosette']; single=sweeps['single_bank_sweeps']; ids=np.ravel(single['bank_id']); sig=np.ravel(single['ref_signal']); z=np.ravel(single['raw_readout']); print('robin',s['robin_x'],s['robin_y'],'sweep raw rows',np.sort(np.max(abs(Y),axis=1))[-10:]);
for k in range(4):
 a=ids==k; xx=sig[a]; yy=z[a]; order=np.argsort(xx); print('single',k,len(xx),np.round(np.c_[xx,yy][order][::max(1,len(xx)//12)],5).tolist())
 T=np.zeros((sum(a),4)); T[:,k]=xx; x=np.r_[X,T]; y=np.r_[Y[:,k],yy]; keep=abs(y)<5; x=x[keep]; y=y[keep]; idx=np.flatnonzero(keep)
 for degree in (1,2,3,4,5):
  terms=[t for d in range(degree+1) for t in itertools.combinations_with_replacement(range(4),d)]; D=np.column_stack([np.prod(x[:,t],axis=1) if t else np.ones(len(y)) for t in terms]); e=np.empty(len(y))
  for f in range(5):
   mask=idx%5!=f; init=np.linalg.lstsq(D[mask],y[mask],rcond=1e-10)[0]; fit=least_squares(lambda b:D[mask]@b-y[mask],init,jac=lambda b:D[mask].copy(),loss='soft_l1',f_scale=.001,max_nfev=60).x; e[~mask]=D[~mask]@fit-y[~mask]
  fit=np.linalg.lstsq(D,y,rcond=1e-10)[0]; print('degree',degree,'CV mixed median/90/max',np.round(np.quantile(abs(e[idx<len(X)]),[.5,.9,1]),6),'single',np.round(np.quantile(abs(e[idx>=len(X)]),[.5,.9,1]),6),'top',[(terms[i],round(fit[i],5)) for i in np.argsort(abs(fit))[-10:][::-1]] if degree<=3 else '')
