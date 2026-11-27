from chamber_probe import *
from scipy.optimize import least_squares
X=m._features(d,d)[:len(yd)]; lab=d['lab_id']; miss=np.isnan(d['coupon_features_raw']).any(axis=1); r=th[2]+X@th[3:]; _,H,_=inp_d
B=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.])); R=np.c_[yd,rd]-H[:,None]*(B[:,:2]@th[:2]); D=r[:,None,None]*B

def fit(keep):
 L=D[keep].reshape(-1,3); y=R[keep].ravel(); p=np.linalg.lstsq(L,y,rcond=None)[0]
 return least_squares(lambda p:L@p-y,p,loss='soft_l1',f_scale=.003).x

def stats(E):
 for name,k in [('all',np.ones(len(E),bool)),('complete',~miss),('imputed',miss)]+[(f'lab{g}',lab==g) for g in np.unique(lab)]:
  print(name,'worst horizon / rate / maximum',np.round([np.max(np.sqrt(np.mean(E[k,:-1]**2,axis=0))),np.sqrt(np.mean(E[k,-1]**2)),np.max(abs(E[k]))],6))
for mode in ('complete','robust-all'):
 sel=~miss if mode=='complete' else np.ones(len(R),bool); E=np.empty_like(R)
 for g in np.unique(lab):
  keep=(lab!=g)&sel; p=fit(keep); E[lab==g]=R[lab==g]-D[lab==g]@p; print(mode,g,p.round(6))
 print(mode,'all-fit',fit(sel)); stats(E)
print('rate errors for imputed diagnostic coupons:',(R-D@fit(~miss))[miss,-1]); print('present feature counts',(~np.isnan(d['coupon_features_raw'])).sum(axis=0)); print('test feature missing counts',np.isnan(te['coupon_features_raw']).sum(axis=0))
