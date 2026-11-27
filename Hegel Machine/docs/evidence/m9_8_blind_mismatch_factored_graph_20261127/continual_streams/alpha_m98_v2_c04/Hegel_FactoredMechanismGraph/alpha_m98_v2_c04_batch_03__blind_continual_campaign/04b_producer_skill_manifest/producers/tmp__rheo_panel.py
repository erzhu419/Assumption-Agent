from rheo_probe import *
from scipy.optimize import least_squares
L=g['lab_id']; G=base-(th[2]+xg@th[3:])[:,None]*t; R=rbase-(th[2]+xg@th[3:]); B=np.c_[yc-G,yr-R]; rawg=g['coupon_features_raw']; sg=g['stratum']; st=np.r_[tr['stratum'],te['stratum']]; rg=g['ref_coupons'];
k=least_squares(lambda q: ((B-(B@f(q)/(f(q)@f(q)))[:,None]*f(q)).ravel()),[1.]) if False else None
def f(q): return np.r_[-np.expm1(-q[0]*t)/q[0],1.]
k=least_squares(lambda q:(B-(B@f(q)/(f(q)@f(q)))[:,None]*f(q)).ravel(),[1.]).x
amp=B@f(k)/(f(k)@f(k)); print('sat',k,'factor RMS',np.sqrt(np.mean((B-amp[:,None]*f(k))**2)))
for l in np.unique(L):
 j=L==l; z=rawg[j]; full=~np.isnan(z).any(1); q=z[full]@th[3:]; target=amp[j][full]; print('lab',l,'raw weighted, target',np.round(np.c_[q,target],4),'fit',np.linalg.lstsq(np.c_[q,np.ones(len(q))],target,rcond=None)[0]);
for groups in ['train','test','both','all','diag','row','diag_first']:
 x2=rawg.copy(); X=raw.copy();
 for l in np.unique(L):
  m=lab==l; j=L==l; z={'train':tr['ref_coupons'][tr['lab_id']==l],'test':te['ref_coupons'][te['lab_id']==l],'both':ref[m],'all':np.r_[ref[m],rg[j]],'diag':rg[j],'row':rg[j],'diag_first':rg[j]}[groups];
  if groups=='row': mu=np.mean(z,1)[:,None]; sd=np.std(z,1)[:,None]
  elif groups=='diag_first': mu=np.median(z); sd=np.std(z)
  else: mu=np.median(z.mean(0)); sd=np.median(z.std(0));
  if np.min(sd)<.01: sd=np.median(ref[m].std(0))
  x2[j]=(x2[j]-mu)/sd
 for fillmode in ['bas','diag','all','zero']:
  XX,LL,SS={'bas':(x,lab,st),'diag':(x2,L,sg),'all':(np.r_[x,x2],np.r_[lab,L],np.r_[st,sg]),'zero':(x,lab,st)}[fillmode]; _,idx,cnt=np.unique(np.c_[LL,SS],axis=0,return_inverse=True,return_counts=True); w=1/cnt[idx,None]; fill=np.nansum(w*XX,0)/np.sum(w*np.isfinite(XX),0); fill=fill if fillmode!='zero' else 0; dd=th[2]+np.nan_to_num(x2,nan=0)@th[3:]+np.isnan(x2)@(np.broadcast_to(fill,4)*th[3:]); err=amp-dd; print(groups,fillmode,round(np.std(err),4),'lab',np.round([np.sqrt(np.mean(err[L==l]**2)) for l in np.unique(L)],4))
