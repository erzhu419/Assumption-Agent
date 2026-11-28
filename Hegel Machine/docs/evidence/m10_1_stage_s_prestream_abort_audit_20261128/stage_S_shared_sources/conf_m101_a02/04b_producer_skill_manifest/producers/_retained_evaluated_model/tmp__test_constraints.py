exec(open('/tmp/discover.py').read().split("print('B',B")[0])
n=len(X); ix=np.arange(n); rng=np.random.default_rng(19); radius=np.linalg.norm((X-X.mean(0))/X.std(0),axis=1); folds=[('tail',ix[:56],ix[56:]),('edge',np.argsort(radius)[:-16],np.argsort(radius)[-16:])]; folds += [('fold'+str(k),np.setdiff1d(ix,j),j) for k,j in enumerate(np.array_split(rng.permutation(n),6))]
u,sv,vt=np.linalg.svd(Z,full_matrices=False); H0=ls(z,Z); print('zero intercept trajectory RMS',np.std(z@H0-Z),'per horizon mean residual',(z@H0-Z).mean(0)); print('affine trajectory intercept',H[0]); print('scalar coefficients on tied products without intercept',ls(np.c_[z,p+q],y)); print('telemetry',s['telemetry_vec']); print('temporal projection bias by horizon',np.std(F@H@(np.eye(len(t))-vt[:2].T@vt[:2]),axis=0))
for name,i,j in folds:
 sy={'full':P[j]@ls(P[i],y[i]),'tied':fs['tied_pair'][j]@ls(fs['tied_pair'][i],y[i]),'origin':np.c_[z[j],p[j]+q[j]]@ls(np.c_[z[i],p[i]+q[i]],y[i]),'fixed':z[j]@ls(z[i],(y-.18*(p+q))[i])+.18*(p[j]+q[j])}
 tz={'affine':F[j]@ls(F[i],Z[i])};
 for nm,D in [('aff',F),('origin',z)]:
  pred=D[j]@ls(D[i],Z[i]); _,_,vh=np.linalg.svd(D[i]@ls(D[i],Z[i]),full_matrices=False); Q=vh[:2].T; tz[nm+'rank2']=pred@Q@Q.T; tz[nm+'rises']=pred@temp([.42,1.8])@np.linalg.pinv(temp([.42,1.8]));
 print(name,'scalar',[(k,round(np.mean(abs(v-y[j])),6)) for k,v in sy.items()],'traj',[(k,*[round(a,6) for a in (np.mean(abs(v-Z[j])),np.mean(abs(v[:,-1]-Z[j,-1])),np.max(np.mean(abs(v-Z[j]),axis=0)))]) for k,v in tz.items()])
