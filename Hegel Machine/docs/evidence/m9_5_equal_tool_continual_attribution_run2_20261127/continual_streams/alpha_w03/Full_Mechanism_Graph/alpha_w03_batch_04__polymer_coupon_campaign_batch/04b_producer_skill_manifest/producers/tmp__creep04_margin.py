exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
from itertools import combinations
# Use complete-feature coupons to separate constitutive residuals from missing-feature drift.
# Probe fixed alternatives, estimating all additional amplitudes off the held-out lab.
scale=d['yield_stress']; stress=h/scale
pairs=np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(1,5),2)])
rat=np.column_stack((stress/(1+stress**2), hold/(1+stress**2),work/(1+stress**2)))
extras={'interactions':pairs,'rational stress':rat}
for ell in np.unique(lab):
 a=(lab!=ell)&~miss; b=(lab==ell)&~miss
 tau,w=s._fit(x[a],h[a],p[a],dt,t,hold[a],work[a],y[a],r[a]); c,v,*_=s._design(tau,x,h,p,dt,t,hold,work)
 ey=y-c@w; er=r-v@w
 good=np.abs(ey[a])<.04; rr=np.abs(er[a])<.04
 print('lab',ell,'baseline median abs curve/rate',np.round([np.median(abs(ey[b])),np.median(abs(er[b]))],6))
 for name,g in extras.items():
  gcurve=g[:,None,:]*t[None,:,None]
  coef=np.linalg.lstsq(np.vstack((gcurve[a][good],g[a][rr])),np.r_[ey[a][good],er[a][rr]],rcond=None)[0]
  print(' ',name,'median abs curve/rate',np.round([np.median(abs(ey[b]-gcurve[b]@coef)),np.median(abs(er[b]-g[b]@coef))],6))
 print('  signed median rate residual by stratum',[(int(j),round(float(np.median(er[b & (tr['stratum']==j)])),6)) for j in np.unique(tr['stratum'][b])])
# Replay occasional interference on the same real spots used in a sparse check.
cal=d['calibration_load_cells']; full={i:s._checks(cal,i) for i in range(4)}
ref={i:s._supported_line(*xy) for i,xy in full.items()}
for m in (1,2):
 errors=[]
 for ell,(xx,yy) in full.items():
  for ids in combinations(range(len(xx)),m):
   if m==2 and np.ptp(xx[list(ids)])<.5: continue
   for idx in range(m):
    for spike in (-.35,.35):
     checks=full.copy(); cx,cy=xx[list(ids)],yy[list(ids)].copy(); cy[idx]+=spike; checks[ell]=(cx,cy)
     fits=ref.copy(); fits[ell]=None
     prior=s._archive_calibration(checks,fits)[ell]
     delta=s._sparse_line(cx,cy,prior)-ref[ell]
     errors.append((float(np.max(abs(delta[0]*np.array([-3.,3.])+delta[1]))),ell,ids,idx,spike))
 print('sparse EMI spots',m,'worst (stress error, lab, row ids, contaminated index, spike)',max(errors),'above 0.005',sum(z[0]>.005 for z in errors),'/',len(errors))
