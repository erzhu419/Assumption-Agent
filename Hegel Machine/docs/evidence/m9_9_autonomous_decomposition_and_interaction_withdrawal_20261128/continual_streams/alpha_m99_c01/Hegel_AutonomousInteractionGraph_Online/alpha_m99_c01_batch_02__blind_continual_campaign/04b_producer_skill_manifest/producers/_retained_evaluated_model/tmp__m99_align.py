from m99_probe import *
from m99_probe import _design
from itertools import combinations
from scipy.optimize import least_squares
O=np.c_[Y,R]; lab=dg['lab_id']; missing=np.isnan(dg['coupon_features_raw']); good=~missing.any(1)
pt,ht=loads(te)
_,b0=_design(tau,xd,pd,hd,dt,t); r0=b0@beta
print('rate residual by row',np.c_[lab,missing.any(1),R-r0].round(4));print('file log keys',[(k,str(v)[:160]) for k,v in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))[-1].items()]);
def normalization(parts):
 rr=np.vstack([p['ref_coupons'] for p in parts]); labs=np.concatenate([p['lab_id'] for p in parts]); a=dg['coupon_features_raw'].copy()
 for l in np.unique(lab):
  ref=rr[labs==l];a[lab==l]=(a[lab==l]-np.median(ref.mean(0)))/np.median(ref.std(0))
 a[missing]=np.broadcast_to(np.mean(np.vstack([x,xt]),axis=0),a.shape)[missing]; return a
for ps in [[tr,te],[dg],[tr,dg],[te,dg],[tr,te,dg]]:
 a=normalization(ps); d=beta[2]+a@beta[3:];gap=b0[:,:2]*tau
 print('ref lens',[len(p['lab_id']) for p in ps], 'rate err complete/missing',*[np.sqrt(np.mean(e**2)) for e in [(gap@ (beta[:2]/tau)+d-R)[good],(gap@(beta[:2]/tau)+d-R)[~good]]]);
 # Multichannel regression isolates temporal fingerprints of the known hereditary variables.
 U=np.c_[np.ones(len(lab)),hd,gap,d]; B=np.linalg.lstsq(U[good],O[good],rcond=None)[0]; E=U@B-O
 print('max lab strain/rate',np.max([np.sqrt(np.mean(E[good&(lab==l)]**2,axis=0)) for l in np.unique(lab)],axis=0)[[0,5,11,12]].round(5))
 print('coeff(t,.25,1.95,4,rate)',B[:,[0,5,11,12]].round(5))
 if len(ps)==1:
  print('diag only rate residual', (gap@(beta[:2]/tau)+d-R).round(5))
  for idx in [good,np.ones(len(lab),dtype=bool)]:
   def proj(logt,out=False):
    ts=np.exp(logt); K=np.r_[np.c_[-np.expm1(-t[:,None]/ts),t],np.r_[1/ts,1][None,:]]; M=(U[:,None,:,None]*K[None,:,None,:]).reshape(len(lab),len(t)+1,-1); c=np.linalg.lstsq(M[idx].reshape(-1,M.shape[-1]),O[idx].ravel(),rcond=None)[0]; E=M@c-O; return (ts,c.reshape(len(U.T),3),E) if out else E[idx].ravel()
   q=least_squares(proj,np.log([.63,1.15]),max_nfev=80);ts,c,E=proj(q.x,True);print('modal',ts,c.round(5),'rms',np.sqrt(np.mean(E[idx]**2,0))[[0,5,11,12]].round(5))
