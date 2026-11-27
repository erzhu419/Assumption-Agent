import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_factor import *
U=np.eye(4)[labs]; sw=cal['single_channel_sweeps']; sweep=np.array([np.polyfit(sw['cmd_stress'][sw['lab_id']==j],sw['true_stress'][sw['lab_id']==j],1) for j in range(4)])[labs]; g,bias=sweep.T
D0=D.copy(); D1=D.copy(); D2=D.copy();
for l in range(4):
 m=labs==l; refs=dg['ref_coupons'][m]; rs=refs.std(0); rm=refs.mean(0); V=dg['coupon_features_raw'][m]; D1[m]=np.where(np.isnan(V),D[m],(V-np.median(rm))/np.median(rs)); D2[m]=np.where(np.isnan(V),D[m],(V-np.mean(rm))/np.mean(rs))

def run(D,mode,adjust):
 r=th[2]+D@th[3:]; h=r-th[2]; rawnan=np.isnan(dg['coupon_features_raw']);
 u={'affine':np.c_[r,np.ones(len(r))],'labs':np.c_[r,U],'sweep':np.c_[r,np.ones(len(r)),g,bias],'gain':np.c_[r,g*r,np.ones(len(r)),g], 'separate':np.c_[np.ones(len(r)),D], 'nonlinear':np.c_[np.ones(len(r)),r,r*r,r*r*r], 'featuretanh':np.c_[np.ones(len(r)),np.tanh(D)@th[3:]], 'labgain':np.c_[U,U*r[:,None]],'calibrated':np.c_[g,h,g*h,bias,np.ones(len(r))]}[mode];
 def fit(ii):
  def obj(p,ret=False):
   M=zstate(P,dt,taus*np.exp(p[:2])); lam=np.exp(p[2]); dk=np.r_[-np.expm1(-t*lam),lam]; Q=np.stack([v[:,None]*dk for v in u.T],-1); base=Y0+M[:,0,None]*th[0]*E[:,0]+M[:,1,None]*th[1]*E[:,1];
   if adjust: Q=np.concatenate((Q,np.stack([v[:,None]*dk for v in rawnan.T],-1)),axis=2)
   A=(Q[ii]*W[None,:,None]).reshape(-1,Q.shape[2]); c=np.linalg.lstsq(A,(base[ii]*W).ravel(),rcond=None)[0]; e=Q@c-base; return (p,c,e) if ret else (e[ii]*W).ravel()
  p=least_squares(obj,[0,-0.8,0],bounds=(-3,3),max_nfev=70).x;return obj(p,True)
 p,c,e=fit(ids); ee=np.empty_like(e);
 for l in np.unique(labs):
  m=labs==l; ee[m]=fit(ids[~m])[2][m]
 def met(q):return np.sqrt(np.mean(q**2,axis=0))[[0,7,11,12]].round(3)
 return met(e),np.max([met(ee[labs==l]) for l in np.unique(labs)],axis=0),np.exp(p).round(3),c.round(3),e
for k,Dx in enumerate([D0,D1,D2]):
 for mode in ['affine','labs','sweep','gain','separate','nonlinear','featuretanh','labgain','calibrated']:
  q=run(Dx,mode,False); print(k,mode,*q[:4]);
  if mode in ('labs','sweep') and k<2: print('rows',q[-1][:,[11,12]].round(3).tolist())
print('telemetry',te['aux_chamber_telemetry'][:10].round(3)); print('telemetry lab medians',[np.median(te['aux_chamber_telemetry'][te['lab_id']==l],axis=0).round(3).tolist() for l in range(4)])
