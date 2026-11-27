exec(open('/tmp/chamber_probe.py').read().split('for name,xc')[0])
Y=np.c_[c['instant_rate_obs'],c['strain_curves_obs']-c['strain_0'][:,None]]
h=c['hold_channel_raw'].sum(-1); labs=c['lab_id']; miss=np.isnan(c['coupon_features_raw']).any(1)
A=lambda s: np.r_[1/s,-np.expm1(-t/s)]
base=h[:,None]*(theta[:2]@np.array([A(s) for s in tau])); E=Y-base
B=np.r_[1,2*t/(np.sqrt(1+t)+1)]; d0=E@B/(B@B)
print('sqrt projection curve/rate residuals',np.sqrt(np.mean((E-d0[:,None]*B)**2,axis=0)))
for name,xc in [('diag',_features(c,c)[0]),('train',_features(a,c)[1]),('test',_features(b,c)[1])]:
 d=theta[2]+xc@theta[3:]
 err=base+d[:,None]*B-Y
 print(name,'lab curve/rate',[(int(l),*[round(float(np.sqrt(np.mean(v**2))),6) for v in (err[labs==l,1:],err[labs==l,0])]) for l in np.unique(labs)],'amplitude errors',np.round(d-d0,4))
 if name=='diag':
  for q in [~miss,miss]: print('miss slice',np.sqrt(np.mean(err[q,1:]**2)),np.sqrt(np.mean(err[q,0]**2)))
  for l in np.unique(labs):
   j=labs==l
   print('lab',l,'d diag/inferred',np.round(np.c_[d[j],d0[j]],4))
# The physical branch weights and times are constrained by benchtop, but test sensitivity
# of the inferred chamber shape and lab standardization to that fit.
for l in np.unique(labs):
 sel=labs!=l
 F=np.c_[h,d0]
 print('holdout shape',l,np.round(np.linalg.lstsq(F[sel],E[sel],rcond=None)[0][1]-B,5))
