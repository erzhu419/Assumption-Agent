exec(open('/tmp/m99_joint.py').read().split('models={')[0])
B=np.r_[np.c_[-np.expm1(-t[:,None]/[.50223,2.52917]),t],[[1/.50223,1/2.52917,1]]]; Z=np.linalg.lstsq(B,Y.T,rcond=None)[0].T
print('lab,stratum,missing,hold,D,Afast,Aslow,drift'); print(np.c_[lab,di['stratum'],np.isnan(di['coupon_features_raw']).sum(1),h,D,Z].round(4))
for mode,xx in [('joint',d[:,1:]),('diag only',m._features(di)),('train+diag',m._features(tr,di)[-N:])]:
 dd=th[2]+xx@th[3:]; print(mode)
 for name,inputs in [('affine',(F(one,h,dd),)*3),('quadratic D',(F(one,h,dd,dd**2),)*3),('saturated D',(F(one,h,dd,np.tanh(dd)),)*3)]:
  r=fit(inputs,print_co=True); rr=r.copy()
  for g in groups: rr[g]=fit(inputs,np.setdiff1d(np.arange(N),g))[g]
  print(name,fmt(r),[fmt(rr[g]) for g in groups])
