exec(open('/tmp/chamber_probe.py').read().split('for tag,s,h')[0])
Y=v['strain_curves_obs']-v['strain_0'][:,None]; R=v['instant_rate_obs']; n=len(R); lab=v['lab_id']; strata=v['stratum']
B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]])
Q=np.c_[Y,R]; A=np.linalg.lstsq(B,Q.T,rcond=None)[0].T
F=lambda parts:{k:np.concatenate([p[k] for p in parts]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
variants={'joint':xv,'local':m._features(v,v)[:n],'train+diag':m._features(tr,v)[-n:],'test+diag':m._features(te,v)[-n:]}
base=np.c_[hv[:,None]*theta[:2],np.zeros(n)]; print('diag rows sorted by inferred drift: drift variants, missing count, amplitude residual')
rows=np.c_[*[xx@theta[2:] for xx in variants.values()],np.isnan(v['coupon_features_raw']).sum(1),A-base]; print(rows[np.argsort(rows[:,0])].round(4))
print('reference shapes',[p['ref_coupons'].shape for p in (tr,te,v)]); print('relative drift output baselines',theta)
for tag,xx in variants.items():
 D=xx@theta[2:]; pred_amp={}
 grammar={'lin':[D],'square':[D,D**2],'abs':[D,abs(D)],'cubic':[D,D**3],'poly3':[D,D**2,D**3],'cross':[D,D*hv],'cross+sq':[D,D*hv,D**2],'quad_cross':[D,D*hv,D**2,D**2*hv],'tanh':[D,np.tanh(D)],'sat':[D,D/(1+D**2)],'Hsq':[D,hv**2],'Xquad':[xx[:,1:],D**2],'Xlin':[xx[:,1:]],'timeH':[D,D**2,np.maximum(D,0)]}
 for name,cols in grammar.items():
  X=np.column_stack([np.ones(n),hv,*cols]); E=[]
  for ii in [lab==k for k in np.unique(lab)]+[strata==k for k in np.unique(strata)]:
   pred=X[ii]@np.linalg.lstsq(X[~ii],Q[~ii],rcond=None)[0]; err=pred-Q[ii]; E.append([np.sqrt(np.mean(err[:,:-1]**2)),np.max(abs(err[:,:-1])),np.sqrt(np.mean(err[:,-1]**2)),np.max(abs(err[:,-1]))])
  pred_amp[name]=np.max(E,axis=0)
 print(tag,[(k,np.round(val,4).tolist()) for k,val in sorted(pred_amp.items(),key=lambda kv:kv[1][0])[:9]])
