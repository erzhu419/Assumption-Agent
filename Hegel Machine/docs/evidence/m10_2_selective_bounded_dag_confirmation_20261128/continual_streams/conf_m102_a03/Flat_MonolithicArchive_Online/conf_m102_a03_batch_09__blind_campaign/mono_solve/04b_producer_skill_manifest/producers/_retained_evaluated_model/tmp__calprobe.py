exec(open('/tmp/probe.py').read().split('m=get(2);')[0])
u=s['panel_a']['u']; v=s['panel_a']['v']; ok=np.linalg.norm(v-u,axis=1)<4*np.median(np.linalg.norm(v-u,axis=1)); print('good',ok.sum()); np.set_printoptions(precision=5,suppress=True,linewidth=180)
for d in (1,2,3):
 keys=[ij for h in range(1,d+1) for ij in combinations_with_replacement(range(4),h)]; F=np.column_stack([np.ones(len(u))]+[np.prod(u[:,ij],axis=1) for ij in keys]); C=fit(F[ok],v[ok]); print('deg',d,'rms',np.mean((F[ok]@C-v[ok])**2)**.5); print('large terms',[(ij,np.round(row,4)) for ij,row in zip([()]+keys,C) if np.max(abs(row))>.01]); print('resid',np.quantile(np.linalg.norm(F@C-v,axis=1),[0,.5,.8,.9,1]))
for j in range(4):
 a=u[:,j]; b=u[:,(j+1)%4]; c=u[:,(j+2)%4]; d=u[:,(j+3)%4]; F=np.c_[np.ones(len(a)),a,b,c,d,a*b,a*c,a*d,a**3,b**3,c**3,d**3,np.tanh(a),np.sin(a),np.tanh(c),np.sin(c)]; C=fit(F[ok],v[ok,j]); print('j',j,np.round(C,5),np.mean((F[ok]@C-v[ok,j])**2)**.5)
