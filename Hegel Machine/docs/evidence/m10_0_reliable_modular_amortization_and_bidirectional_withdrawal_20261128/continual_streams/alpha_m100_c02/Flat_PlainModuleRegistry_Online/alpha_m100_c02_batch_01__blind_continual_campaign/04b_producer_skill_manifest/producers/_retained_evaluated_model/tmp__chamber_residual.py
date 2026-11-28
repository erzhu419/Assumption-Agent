from chamber_probe import *
xd=m._features(g,g)[:len(h)]; dr=xd@th[2:]; b=basis(tau); o=n.ones(len(h)); lab=g['lab_id']; mis=n.isnan(g['coupon_features_raw']);
# Fit the load response at its material value; this isolates the chamber correction.
base=h[:,None]*(b[:,:2]@th[:2]); target=y-base
F=(n.c_[o,dr],n.c_[o,dr],n.c_[o,dr]); A=n.column_stack([(f[:,None,:]*b[None,:,j,None]).reshape(y.size,-1) for j,f in enumerate(F)])
q=least_squares(lambda q:A@q-target.ravel(),n.linalg.lstsq(A,target.ravel(),rcond=None)[0],loss='soft_l1',f_scale=.003).x
err=(A@q).reshape(y.shape)-target
print('correction',q); print('lab, drift, missing, hold, residual first/middle/last/rate'); print(n.c_[lab,dr,mis,h,err[:,[0,5,-2,-1]]].round(4));
for group in [tr,te,g]:
 print('panel stats',[(i,round(n.median(a.mean(0)),4),round(n.median(a.std(0)),4)) for i in n.unique(group['lab_id']) for a in [group['ref_coupons'][group['lab_id']==i]]]); print('features first',group['coupon_features_raw'][:2].round(3));
print('ranges test diag hold',m._loads(c,te)[1].min(),m._loads(c,te)[1].max(),h.min(),h.max())
