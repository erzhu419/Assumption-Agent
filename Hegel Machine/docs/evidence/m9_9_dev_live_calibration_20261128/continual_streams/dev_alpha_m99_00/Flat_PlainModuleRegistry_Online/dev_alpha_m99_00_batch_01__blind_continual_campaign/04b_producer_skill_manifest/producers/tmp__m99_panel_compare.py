import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/m99_probe.py').read())
Y=np.c_[ch['strain_curves_obs']-ch['strain_0'][:,None],ch['instant_rate_obs']]; K=np.r_[np.log1p(t),1.]; H=np.r_[-np.expm1(-t[:,None]/tau)@theta[:2],theta[:2]@(1/tau)]; lab=ch['lab_id']; flags=np.isnan(ch['coupon_features_raw']).any(axis=1)
for splits in [(ch,), (tr,ch),(te,ch),(tr,te,ch)]:
 Q=m._features(*splits)[-len(lab):]; a=theta[2]+Q@theta[3:]; E=Y-cc[1][:,None]*H-a[:,None]*K; print('panels',len(splits),[len(w['lab_id']) for w in splits], 'rmse',np.sqrt(np.mean(E**2)),'max',abs(E[:,:-1]).max(),abs(E[:,-1]).max()); print('by lab complete/missing',[(int(j),round(np.std(E[(lab==j)&~flags]),5),round(np.mean(E[(lab==j)&~flags]),5)) for j in np.unique(lab)])
for name,w in [('tr',tr),('te',te),('ch',ch)]:
 p=w['ref_coupons']; print(name,'panels');
 for j in np.unique(w['lab_id']):
  q=p[w['lab_id']==j];print(j,len(q),np.median(q.mean(0)),np.median(q.std(0)),np.round(np.median(q,axis=0),2))
