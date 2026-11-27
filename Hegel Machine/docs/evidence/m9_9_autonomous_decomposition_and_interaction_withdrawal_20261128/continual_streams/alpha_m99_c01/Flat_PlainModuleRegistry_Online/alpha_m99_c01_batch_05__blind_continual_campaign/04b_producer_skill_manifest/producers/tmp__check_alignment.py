exec(open('/tmp/chamber_amplitude.py').read().split('zsets=')[0].replace('print(np.c_[C,H,G,D,U].round(4))','pass'))
import downstream_solver as dn

def compare(train,test,which):
 raw=np.vstack([train['coupon_features_raw'],test['coupon_features_raw']]).copy(); ref=np.vstack([train['ref_coupons'],test['ref_coupons']]); labs=np.r_[train['lab_id'],test['lab_id']]; strata=np.r_[train['stratum'],test['stratum']]
 for ll in np.unique(labs):
  j=labs==ll; raw[j]=(raw[j]-np.median(ref[j].mean(0)))/np.median(ref[j].std(0))
 _,i,n=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); w=1/n[i]; ok=np.isfinite(raw)
 if which=='balanced': center=np.nansum(raw*w[:,None],0)/(ok.T@w)
 elif which=='mean': center=np.nanmean(raw,0)
 elif which=='observed-balanced': center=np.array([np.sum([np.nanmean(raw[i==k,j]) for k in np.unique(i[ok[:,j]])])/len(np.unique(i[ok[:,j]])) for j in range(raw.shape[1])])
 filled=np.where(ok,raw,center)
 return np.split(filled,[len(train['lab_id'])]),center
for mode in ('balanced','mean','observed-balanced'):
 (xx,_),ct=compare(a,b,mode); (_,qc),cc=compare(c,c,mode); tc,bc=m._fit(xx,p,h,dt,t,y,r); aa,br=m._design(tc,xx,p,h,dt,t); et=aa@bc-y; er=br@bc-r
 complete=np.isfinite(c['coupon_features_raw']).all(1); train_complete=np.isfinite(a['coupon_features_raw']).all(1)
 basis=np.vstack((np.c_[-np.expm1(-t[:,None]/tc),t],np.r_[1/tc,1.])).T; design=(bc[2]+qc@bc[3:])[:,None,None]*basis.T; base=np.c_[H[:,None]*bc[:2],np.zeros(len(H))]@basis; target=np.c_[Y,R]-base
 err=np.empty_like(target)
 for ll in np.unique(lab):
  s=complete&(lab!=ll); k=np.linalg.lstsq(design[s].reshape(-1,3),target[s].ravel(),rcond=None)[0]; err[lab==ll]=design[lab==ll]@k-target[lab==ll]
 rms=lambda z:np.sqrt(np.mean(z**2))
 print(mode,'centers',ct.round(5),cc.round(5),'tau',tc.round(6),'train slice median',[(np.median(abs(et[s])),np.median(abs(er[s]))) for s in (train_complete,~train_complete)],'diagnostic OOF strain/rate',[(rms(err[s,:-1]),rms(err[s,-1])) for s in (complete,~complete)])
print('public item count',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
