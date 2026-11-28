exec(open('/tmp/readout_diag.py').read().split('for keys in')[0])
_,(b,h),_,_=m._features(s,x,unmix=True)
folds += [(np.argsort(sign*z[:,j])[:-10],np.argsort(sign*z[:,j])[-10:]) for j in range(4) for sign in (1,-1)]
for config in [(4,.15,3),(4,.15,1),(4,.15,8),(3,.4,1),(3,.4,3),(3,.15,15),(5,.05,20)]:
 errors=[]
 for it,iv in folds:
  ix=np.r_[it,iv]; D=m._polynomial_features(tuple(k[ix] for k in (z,b,h)),len(it),*config)
  a=sink.run_node({'node_00':{'g':D,'fd':b[ix],'cd':np.c_[c1,c2][ix],'y':y[it],'traj':tr[it]}})
  errors.append([np.mean(abs(a['y_hat']-y[iv])),np.mean(abs(a['state_pred']-tr[iv]))])
 E=np.array(errors); print(config,'official',E[0].round(5),'random max',E[:5].max(0).round(5),'tail max',E[5:].max(0).round(5),'tail mean',E[5:].mean(0).round(5))
