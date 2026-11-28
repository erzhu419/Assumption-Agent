import numpy as np, pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; single=sw['single_bank_sweeps']; terms=[t for d in range(4) for t in combinations_with_replacement(range(4),d)]
def design(x,terms):
 return np.column_stack([np.prod(x[:,t],axis=1) if t else np.ones(len(x)) for t in terms])
for j in range(4):
 m=single['bank_id']==j; solo=np.zeros((m.sum(),4)); solo[:,j]=single['ref_signal'][m]; xx=np.vstack([x,solo]); yy=np.r_[y[:,j],single['raw_readout'][m]]; keep=np.abs(yy)<6; D=design(xx,terms)
 for _ in range(5):
  c=np.linalg.lstsq(D[keep],yy[keep],rcond=None)[0]; r=yy-D@c; keep=np.abs(r)<max(1e-4,8*np.median(abs(r[keep])))
 old=[(),(j,),((j+1)%4,),((j-1)%4,),tuple(sorted([j,(j+1)%4])),(j,)*3]; P=design(xx,old); oldr=yy-P@np.linalg.lstsq(P[keep],yy[keep],rcond=None)[0]; print(j,'full residual p50/pmax', np.median(abs(r[keep])),max(abs(r[keep])), 'previous residual p50/pmax',np.median(abs(oldr[keep])),max(abs(oldr[keep])), 'excluded',np.flatnonzero(~keep)); print([(t,round(a,6)) for t,a in zip(terms,c) if abs(a)>0.0003]); print('nonpolynomial witness', sorted(abs(r[keep]))[-5:])
