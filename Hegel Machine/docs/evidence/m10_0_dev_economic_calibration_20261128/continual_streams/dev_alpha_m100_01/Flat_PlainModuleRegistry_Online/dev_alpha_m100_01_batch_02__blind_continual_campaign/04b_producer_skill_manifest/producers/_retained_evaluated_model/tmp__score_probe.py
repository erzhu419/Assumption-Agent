from chamber_probe import *
past,h=loads(dg); y=obs(dg); ta,th=fit(np.arange(n)); B=basis(ta); labs=np.asarray(dg['lab_id']); N=len(labs)
z=np.zeros((N,2))
for i,v in enumerate(dt): z=np.exp(-v/ta)*z-past[:,i,None]*np.expm1(-v/ta)
s0=th[2]+diagfeat()[1]@th[3:]; e=y-np.c_[(h[:,None]-z)*th[:2],s0]@B.T
print('lab references location, scale; train, test, diagnostic, pooled')
for k in np.unique(labs):
 print(k,[(round(np.median(d['ref_coupons'][d['lab_id']==k].mean(0)),4),round(np.median(d['ref_coupons'][d['lab_id']==k].std(0)),4)) for d in [train,test,dg,{j:np.concatenate([train[j],test[j]]) for j in ['lab_id','ref_coupons']}]])
def cmp(A, label):
 A=A.reshape(N,len(B),-1); met=[]; vals=[]
 for m in [np.zeros(N,bool)]+[labs==j for j in np.unique(labs)]+[np.arange(N)==j for j in range(N)]:
  k=np.linalg.lstsq(A[~m].reshape(-1,A.shape[-1]),e[~m].ravel(),rcond=None)[0]; p=(e-A@k)[m] if m.any() else e-A@k; vals.append(p)
  met.append([np.sqrt(np.mean(p[:,:-1]**2)),np.sqrt(np.mean(p[:,-2]**2)),np.sqrt(np.mean(p[:,-1]**2))]);
 print(label,np.round(met[0],5),np.max(met[1:5],0).round(5),np.sqrt(np.mean(np.square(met[5:]),0)).round(5), 'fit',np.linalg.lstsq(A.reshape(-1,A.shape[-1]),e.ravel(),rcond=None)[0].round(4))
def design(x): return (x[:,None,:,None]*B[None,:,None,:]).reshape(N,len(B),-1)
for label,parts in [('pooled',[train,test]),('train',[train]),('test',[test]),('self',[dg]),('all',[train,test,dg])]:
 r=np.concatenate([d['ref_coupons'] for d in parts]); ll=np.concatenate([d['lab_id'] for d in parts]); x=dg['coupon_features_raw'].copy()
 for j in np.unique(labs):
  rr=r[ll==j]; x[labs==j]=(x[labs==j]-np.median(rr.mean(0)))/np.median(rr.std(0))
 means=[]
 lx=np.r_[train['lab_id'],test['lab_id']]; st=np.r_[train['stratum'],test['stratum']]; wx=np.r_[train['coupon_features_raw'],test['coupon_features_raw']].copy(); rr=np.r_[train['ref_coupons'],test['ref_coupons']]
 for j in np.unique(lx):
  p=rr[lx==j]; wx[lx==j]=(wx[lx==j]-np.median(p.mean(0)))/np.median(p.std(0))
 _,inv,c=np.unique(np.c_[lx,st],axis=0,return_inverse=True,return_counts=True); w=1/c[inv]; mu=np.nansum(wx*w[:,None],0)/np.sum(np.isfinite(wx)*w[:,None],0)
 for name,fill in [('zero',0),('balanced',mu),('self',np.nanmean(x,0))]:
  xx=np.where(np.isfinite(x),x,fill); s=th[2]+xx@th[3:]; d=np.zeros((N,len(B),1)); d[:,:,0]=(s-s0)[:,None]*B[:,2]; ee=e.copy(); e[:]-=d[:,:,0]; cmp(design(np.c_[np.ones(N),s]),label+' '+name); e[:]=ee
print('pooled model expanded:'); cmp(design(np.c_[np.ones(N),s0,s0*s0]),'quadratic'); cmp(design(np.c_[np.ones(N),s0,np.isnan(diagfeat()[0])[:,[0,3]]]),'missing indicator');
A=design(np.c_[np.ones(N),s0]); q=np.linalg.lstsq(A.reshape(-1,6),e.ravel(),rcond=None)[0]; p=e-A@q
print('residual final,rate by lab',np.c_[labs,np.isnan(diagfeat()[0]).sum(1),s0,p[:,-2:]].round(4))
print('test telemetry keys',list(test))
