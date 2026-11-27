exec(open('/tmp/explore98.py').read().split('for name,X in')[0])
spec=m._specimen(g,align(g),bridge); c,q=m._branches(spec,dt,t,tau); elastic=np.column_stack((c@w[:2],q@w[:2])); E=obs-elastic
finite=np.isfinite(g['coupon_features_raw']).all(axis=1)
print('diagnostic kernel, profiled amplitudes by coupon')
for name, form in [('exp', lambda k: -np.expm1(-t*k)/k), ('rational',lambda k:t/(1+k*t))]:
 def fit(rows):
  def res(p):
   kernel=np.r_[form(p[0]),1.]; v=E[rows]; a=(v@kernel)/(kernel@kernel); return (a[:,None]*kernel-v).ravel()
  return least_squares(res,[1.],bounds=(.001,10.),loss='soft_l1',f_scale=.004).x[0]
 k=fit(np.ones(len(r),bool)); preds=[]
 for lab in np.unique(g['lab_id']):
  held=g['lab_id']==lab; kk=fit(~held); e=E[held,-1,None]*form(kk)-E[held,:-1]; preds.append((int(lab),round(kk,6),round(np.sqrt(np.mean(e**2)),6)))
 print(name,k,'held-lab time kernel RMSE using measured initial rates',preds)
for label,Y in [('train+test',align(g)),('train+diagnostics',m._features(tr,g)[1]),('all three',m._features(tr,{k:np.concatenate((te[k],g[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')})[1][-len(r):]),('diagnostics own panel',m._features(g,g)[1])]:
 drift=w[2]+Y@w[3:]; pred=elastic+drift[:,None]*np.r_[-np.expm1(-t),1.]; err=pred-obs
 print(label,stats(pred),'complete vs incomplete RMSE',[(round(np.sqrt(np.mean(err[i,:-1]**2)),6),round(np.sqrt(np.mean(err[i,-1]**2)),6)) for i in (finite,~finite)],'lab rate bias',[(int(lab),round(np.median(err[g['lab_id']==lab,-1]),6)) for lab in np.unique(g['lab_id'])])
