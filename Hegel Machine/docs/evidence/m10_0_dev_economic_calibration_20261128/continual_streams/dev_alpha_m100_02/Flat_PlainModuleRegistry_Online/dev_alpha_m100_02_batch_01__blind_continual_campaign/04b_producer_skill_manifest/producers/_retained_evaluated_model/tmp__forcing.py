exec(open('/tmp/memory.py').read().split("for label, x in")[0])
raw=c['coupon_features_raw']; good=np.isfinite(raw).all(1); y=A[:,:2]-hold[:,None]*w[:2]; base=np.mean(y/np.array([.28,1.9]),axis=1); print('branch correction collinearity',np.linalg.svd(y,compute_uv=False)); print('slow vs fast',np.polyfit(y[:,0],y[:,1],1)); print('drift vs slow',np.polyfit(y[:,1],A[:,2],1));
for label, x in [('train+diagnostic',xc1),('all',xc2),('frozen',xc3),('diagnostic-only',m['_features'](c,c)[0])]:
 r=np.c_[np.ones(len(x)),x]@w[2:]; print(label,'fit complete',np.linalg.lstsq(np.c_[np.ones(good.sum()),r[good]],y[good],rcond=None)[0], 'err corr',np.std(y[good]-np.c_[np.ones(good.sum()),r[good]]@np.linalg.lstsq(np.c_[np.ones(good.sum()),r[good]],y[good],rcond=None)[0],axis=0));
 for label2, F in [('shared drift',np.c_[np.ones(n),r]),('unrestricted features',np.c_[np.ones(n),x]),('squared drift',np.c_[np.ones(n),r,r*r])]:
  print(label2,score(block(F,F,F)) if False else score(block(F,F,F)+0)[0]);
 print('residualized',score(block(F,F,F))[0]) if False else None
# Chamber hold amplitudes are fixed to material values in these comparisons.
Y-=np.c_[hold[:,None]*w[:2],np.zeros(n)]@T.T
for label, x in [('train+diagnostic',xc1),('all',xc2),('frozen',xc3),('diagnostic-only',m['_features'](c,c)[0])]:
 r=np.c_[np.ones(n),x]@w[2:];
 for key,F in [('proportional',r[:,None]),('affine',np.c_[np.ones(n),r]),('independent feature weights',np.c_[np.ones(n),x]),('quadratic',np.c_[np.ones(n),r,r*r])]: print(label,key,score(block(F,F,F)))
print('refs center and scale across pools')
for l in np.unique(lab):
 print(l,[(round(np.median(a['ref_coupons'][a['lab_id']==l].mean(0)),4),round(np.median(a['ref_coupons'][a['lab_id']==l].std(0)),4)) for a in (tr,te,c)])
