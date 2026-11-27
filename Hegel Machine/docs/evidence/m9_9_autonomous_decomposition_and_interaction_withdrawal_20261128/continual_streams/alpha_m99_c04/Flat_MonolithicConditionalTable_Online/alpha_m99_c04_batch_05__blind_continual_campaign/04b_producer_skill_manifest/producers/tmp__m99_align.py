exec(open('/tmp/m99_probe.py').read().split('base=np.column_stack')[0])
from itertools import combinations
labs=d['lab_id']; missing=np.isnan(d['coupon_features_raw']).any(axis=1); obs=np.column_stack([y,r]); w=np.r_[np.full(len(t),1/.004),1/.005]; base=(h[:,None]*theta[:2]) @ A[:,:2].T

def combine(*gs):
 return {k:np.concatenate([g[k] for g in gs]) for k in ('lab_id','stratum','coupon_features_raw')} | {'ref_coupons':np.concatenate([g['ref_coupons'] for g in gs])}
X={'train+diag':xd,'train+test':m._features(combine(tr,te),d)[1], 'diag only':m._features(d,d)[1]}
# Apply the fixed training/deployment imputation means to diagnostics without including them in the panel.
import inspect
print(inspect.getsource(m._features))
Xall=np.vstack([x,xe]); ga=combine(tr,te); _,idx,counts=np.unique(np.column_stack([ga['lab_id'],ga['stratum']]),axis=0,return_inverse=True,return_counts=True); known=np.isfinite(ga['coupon_features_raw']); fill=np.sum(Xall*known/counts[idx,None],axis=0)/np.sum(known/counts[idx,None],axis=0); print('fixed fill',fill)
X['fixed panel']=np.where(np.isnan(d['coupon_features_raw']),fill,xd)

def run(z,names,fitmask):
 feats={'d':z,'hd':h*z,'m0':history[:,0],'m1':history[:,1]}; F=np.column_stack([feats[n] for n in names]); D=np.einsum('ij,kb->ikjb',F,A).reshape(len(h),len(w),-1)
 def fit(mask):
  M=(D[mask]*w[None,:,None]).reshape(-1,D.shape[-1]); v=((obs[mask]-base[mask])*w).ravel(); return least_squares(lambda u:M@u-v,np.linalg.lstsq(M,v,rcond=None)[0],loss='soft_l1',f_scale=2).x
 pred=np.empty_like(obs)
 for k in np.unique(labs): pred[labs==k]=base[labs==k]+D[labs==k]@fit((labs!=k)&fitmask)
 e=np.abs(pred-obs); print(names,'train n',sum(fitmask),'coef',np.round(fit(fitmask),4),'clean MAD',np.median(e[~missing,:-1]),np.median(e[~missing,-1]),'missing MAD',np.median(e[missing,:-1]),np.median(e[missing,-1])); report('LOLO',pred[:,:-1],pred[:,-1])
for label,xx in X.items():
 print(label,'drift on missing',np.round((theta[2]+xx@theta[3:])[missing],5)); z=theta[2]+xx@theta[3:]; run(z,('d',),~missing); run(z,('d','hd'),~missing); run(z,('d',),np.ones(len(z),bool));
print('corrected correlations with d and hd:',np.corrcoef(np.column_stack([drift,h*drift,coeff]).T)[:2,2:]); print('load extrema',np.min(h),np.max(h)); print('diagnostic versus joint missing rows', np.where(missing)[0])
