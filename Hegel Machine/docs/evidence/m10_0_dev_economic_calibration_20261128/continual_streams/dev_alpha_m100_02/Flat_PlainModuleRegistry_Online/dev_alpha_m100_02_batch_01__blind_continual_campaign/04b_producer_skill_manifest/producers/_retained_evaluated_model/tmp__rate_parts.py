exec(open('/tmp/memory.py').read().split('for label, x in')[0])
x=m['_features'](c,c)[0]; r=np.c_[np.ones(n),x]@w[2:]; good=np.isfinite(c['coupon_features_raw']).all(1); Y0=Y-np.c_[hold[:,None]*w[:2],np.zeros(n)]@T.T; TT=T*np.r_[tau,1]; M=r[:,None,None]*TT[None,:,:];
for label,k in [('all',np.ones(n,bool)),('complete',good),*[(f'without {l}',lab!=l) for l in np.unique(lab)]]:
 X=M[k].reshape(-1,3); yy=Y0[k].ravel(); p=np.linalg.lstsq(X,yy,rcond=None)[0]; pr=least_squares(lambda q:X@q-yy,p,loss='soft_l1',f_scale=.004).x; print(label,'rate fractions',p,pr,'sum',p.sum(),pr.sum())
for p in ([.45,.52,.03],[.45,.525,.03],[.45,.52,.035],[.45,.525,.025]):
 e=Y0-M@p; print(p, 'lab RMSE',[(int(l),*np.round([np.mean(e[lab==l,:-1]**2)**.5,np.mean(e[lab==l,-1]**2)**.5],5)) for l in np.unique(lab)], 'complete',np.mean(e[good]**2)**.5)
