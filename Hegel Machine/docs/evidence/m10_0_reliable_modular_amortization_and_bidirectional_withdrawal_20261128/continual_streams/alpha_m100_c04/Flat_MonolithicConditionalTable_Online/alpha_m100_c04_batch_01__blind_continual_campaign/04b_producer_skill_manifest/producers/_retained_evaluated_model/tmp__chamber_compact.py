exec(open('/tmp/chamber_probe.py').read().split('# Regress each component')[0])
L=q['lab_id']; S=q['stratum']; ids=np.arange(len(h)); xq=s._features(q,q)[:len(h)]; D=xq@w[3:]+w[2]; O=np.ones(len(h)); miss=~np.isfinite(q['coupon_features_raw']); C=np.c_[O,h,D];
models={
'linear':[C,C,C], 'quadratic':[np.c_[C,D**2],np.c_[C,D**2],C], 'hinge':[np.c_[C,np.maximum(-D,0)],np.c_[C,np.maximum(-D,0)],C], 'drift_hold':[np.c_[C,D*h],np.c_[C,D*h],C], 'features':[np.c_[O,h,xq]]*2+[C], 'missing':[np.c_[C,miss]]*2+[C], 'lab_gain':[np.c_[C,np.eye(4)[L][:,1:],np.eye(4)[L][:,1:]*D[:,None]]]*2+[C], 'cubic':[np.c_[C,D**2,D**3]]*2+[C]}
def fit(z,ii,G):
 B=basis(z); F=np.concatenate([B[None,:,j,None]*J[:,None,:] for j,J in enumerate(G)],axis=-1)
 b=np.linalg.lstsq(F[ii].reshape(-1,F.shape[-1]),y[ii].ravel(),rcond=None)[0]; return F@b,b
def run(ii,G,free):
 z=least_squares(lambda z:(fit(z,ii,G)[0][ii]-y[ii]).ravel(),[.712,4.8],bounds=([.5,2.5],[1.,10]),max_nfev=45).x if free else tau
 f,b=fit(z,ii,G); return f,z,b
for name,G in models.items():
 for free in (False,True):
  f,z,b=run(ids,G,free); E=[]
  for groups in (ids,L,S):
   e=np.empty_like(y)
   for j in np.unique(groups):
    m=groups==j; e[m]=run(ids[~m],G,free)[0][m]-y[m]
   E.append([np.sqrt(np.mean(e[:,:-1]**2)),np.max(np.sqrt(np.mean(e[:,:-1]**2,axis=0))),np.sqrt(np.mean(e[:,-1]**2))])
  print(name, free, z.round(4),'CV',np.round(E,5).tolist(),'fit',np.sqrt(np.mean((f-y)**2)).round(5),'beta',b.round(4) if name in ('linear','quadratic','hinge','drift_hold') else '')
f,z,b=run(ids,models['linear'],True); print('linear signed residual slices: lab stratum hold drift misses end rate'); print(np.c_[L,S,h,D,miss.sum(1),(f-y)[:,-2:]].round(5))
