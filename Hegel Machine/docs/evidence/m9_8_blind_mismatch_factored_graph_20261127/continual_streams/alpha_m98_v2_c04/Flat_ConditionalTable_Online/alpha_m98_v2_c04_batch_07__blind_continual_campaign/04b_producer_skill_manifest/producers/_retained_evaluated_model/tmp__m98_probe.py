import m98_compare as M
import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/m98_arrays.npz'); globals().update({k:z[k] for k in z.files}); m=M.mods['batch_06']; cal=M.d['calibration_load_cells']; u=cal['true_channel_loads']; v=cal['bridge_raw']; print('sweeps',[(k,type(a).__name__,getattr(a,'shape',None)) for k,a in cal['single_channel_sweeps'].items()]);
for name,a in [('affine',np.broadcast_to(np.c_[np.ones(len(u)),u].T[None,:,:],(4,5,len(u))).transpose(0,2,1)),('archived',m._bridge_basis(u).transpose(1,0,2)),('union',np.concatenate((np.broadcast_to(np.c_[np.ones(len(u)),u][None,:,:],(4,len(u),5)),m._bridge_basis(u).transpose(1,0,2)[:,:,4:]),axis=-1))]:
 e=np.empty_like(v)
 for k in range(5):
  mask=np.arange(len(u))%5==k
  for j in range(4):
   A=a[j,~mask]; y=v[~mask,j]; p=np.linalg.lstsq(A,y,rcond=None)[0]; p=least_squares(lambda p:A@p-y,p,loss='soft_l1',f_scale=.002).x; e[mask,j]=a[j,mask]@p-v[mask,j]
 print('cal',name,np.quantile(abs(e),[.5,.9,.95,.99,1]))
drift=th[2]+x@th[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; gap=hold[:,None]-past@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau)); G=gap*th[:2]; c0=G@(-np.expm1(-t[None,:]/tau[:,None]))+drift[:,None]*t; r0=G@(1/tau)+drift
print('tau',tau,'theta',th); print('ranges hold/gap/drift',np.quantile(np.column_stack((hold,G,drift)),[0,.5,1],axis=0)); print('curve resid per time',np.quantile(yc-c0,[.1,.5,.9],axis=0).round(3)); print('rate resid',np.round(yr-r0,4)); print('curves / baseline sample',np.c_[yc[[0,1,2],:].T,c0[[0,1,2],:].T].round(3));
for name,F in [('chron',np.c_[G,drift]),('hold+chron',np.c_[G,drift,hold]),('hold+drift',np.c_[hold,drift]),('fullhistory',np.c_[hold,past,drift])]:
 B=[]; E=[]
 for y in np.c_[yc,yr].T:
  p=np.linalg.lstsq(F,y,rcond=None)[0]; p=least_squares(lambda p:F@p-y,p,loss='soft_l1',f_scale=.02).x; B.append(p); E.append(np.median(abs(F@p-y)))
 print(name,'med',np.round(E,4),'coeff t0,t6,tlast,rate',np.array(B)[[0,6,-2,-1]].round(4))
