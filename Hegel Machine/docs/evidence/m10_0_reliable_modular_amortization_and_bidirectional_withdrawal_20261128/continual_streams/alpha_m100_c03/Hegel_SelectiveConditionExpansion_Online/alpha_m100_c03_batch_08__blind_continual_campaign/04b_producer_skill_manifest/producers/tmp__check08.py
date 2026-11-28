exec(open('/tmp/probe08.py').read().split('print(\'intrinsic\'')[0])
from itertools import combinations
F=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); L=np.r_[t,1.]; R=F*tau; ref=h[:,None]*(F@th[:2]); target=np.c_[y(q),q['instant_rate_obs']]; basis=np.c_[R[:,0]-L,R[:,1]-L]; labs=q['lab_id']; pte,hte=_loads(te)
for name, ps, hs in [('diag',p,h),('test',pte,hte)]:
 print(name,'hold range',hs.min(),hs.max(),'hold-minus-last',np.quantile(hs-ps[:,-1],[0,.1,.5,.9,1]),'negative counts',np.sum(hs<0),np.sum(hs<ps[:,-1]),'past range',ps.min(),ps.max())
def feat_by(name):
 if name=='train+diag': return xq
 if name=='test+diag': return _features(te,q)[len(hte):]
 if name=='diag': return _features(q,q)[len(h):]
 source=(tr,te,q) if name=='all' else (tr,te)
 v=q['coupon_features_raw'].copy(); ac=np.concatenate([s['lab_id'] for s in source]); panel=np.vstack([s['ref_coupons'] for s in source]); xx=np.vstack([s['coupon_features_raw'] for s in source]); strata=np.concatenate([s['stratum'] for s in source]);
 for lab in np.unique(ac):
  m=ac==lab; panel0=panel[m]; mu=np.median(panel0.mean(0)); sig=np.median(panel0.std(0)); xx[m]=(xx[m]-mu)/sig; v[labs==lab]=(v[labs==lab]-mu)/sig
 _,c,ns=np.unique(np.c_[ac,strata],axis=0,return_inverse=True,return_counts=True); w=1/ns[c,None]; return np.where(np.isnan(v),np.nansum(w*xx,axis=0)/np.sum(w*~np.isnan(xx),axis=0),v)
def stats(pred):
 err=pred-target; slices=[labs==i for i in np.unique(labs)]+[h<p[:,-1],h>=p[:,-1]]
 return np.round([np.sqrt(np.mean(err[:,:-1]**2)), np.sqrt(np.mean(err[:,-1]**2)), max(np.sqrt(np.mean(err[s,:-1]**2)) for s in slices if s.any()),max(np.sqrt(np.mean(err[s,-1]**2)) for s in slices if s.any()),np.max(abs(err[:,-2]))],5)
for name in ('train+diag','test+diag','diag','all','train+test'):
 f=feat_by(name); r=th[2]+f@th[3:]; U=r[:,None,None]*basis; O=ref+r[:,None]*L
 g=np.linalg.lstsq(U.reshape(-1,2),(target-O).ravel(),rcond=None)[0]
 for kind,G in [('conserved',g),('rounded',[.2,.55])]:
  out=O+U@G; print(name,kind,np.round(G,5),stats(out))
 out=np.empty_like(target)
 for lab in np.unique(labs):
  m=labs!=lab; G=np.linalg.lstsq(U[m].reshape(-1,2),(target[m]-O[m]).ravel(),rcond=None)[0]; out[~m]=O[~m]+U[~m]@G
 print(name,'LOLO',stats(out),'rate residual',np.round((target-ref)[:,-1]-r,4))
 # Fit the three amplitudes while allowing the drift standardization for diagnostic coupons to be wrong.
 for G in ([.2,.55],[.2,.5],[.2,.6],[.2,.8],[.5,.5]):
  b=L+basis@G; obs=target-ref; drift=obs@b/(b@b); print('latent drift',G,np.sqrt(np.mean((ref+drift[:,None]*b-target)**2)),np.sqrt(np.mean((drift-r)**2)))
