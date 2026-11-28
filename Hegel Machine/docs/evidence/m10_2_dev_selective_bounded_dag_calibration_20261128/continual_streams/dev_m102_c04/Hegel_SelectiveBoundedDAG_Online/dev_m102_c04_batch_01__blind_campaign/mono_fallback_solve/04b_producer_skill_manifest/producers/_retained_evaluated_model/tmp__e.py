from d import *
u,v=[s['panel_a'][k] for k in ('u','v')]; m=np.linalg.norm(v-u,axis=1)<2
print('calibration retained',m.sum())
w=ls(A(u[m]),v[m]);print(w);print('errors',np.sort(np.linalg.norm(A(u)@w-v,axis=1)));wd=np.array([ls(A(u[m])[:,[0,i+1]],v[m,i]) for i in range(4)]).T;print('diag',wd,np.median(abs(u[m]*wd[1]+wd[0]-v[m]),0));
b=ls(A(s['panel_b']['u']),s['panel_b']['v']);print('B',b); print('time tele',t,s['telemetry_vec']);print('T',T[:8]);print('y',y[:8]);print('x',x[:8]);print('quantiles',np.quantile(np.c_[y,T[:,0],T[:,-1]],np.linspace(0,1,11),axis=0));print('y on traj',np.mean(abs(A(T)@ls(A(T[:52]),y[:52])-y)[52:]));
q=np.linalg.solve(w[1:].T,(x-w[0]).T).T;print('q ranges',q.min(0),q.max(0));z=A(q)@b;Z=np.c_[y,T[:,0],T[:,-1]]
for label,p in [('z',z),('q',q),('x',x)]:
 f=poly(p,2);c=ls(f[:52],Z[:52]);print(label,'poly2\n',c,'\nerrors',np.mean(abs(f@c-Z)[:52],0),np.mean(abs(f@c-Z)[52:],0));
for p in (q,z,np.c_[q,z]):
 FF=[poly(p,2),poly(p,3)];
 for xform in (np.tanh,np.abs,np.sin,lambda a:a/(1+abs(a)),lambda a:a*a/(1+a*a),lambda a:np.sqrt(abs(a))):
  FF += [np.c_[poly(p,2),xform(p)],poly(xform(p),2)]
 for i,f in enumerate(FF):
  f=f[:,np.std(f,0)>1e-10];f=(f-np.mean(f[:52],0))/np.std(f[:52],0);l,v,r=np.linalg.svd(f[:52],full_matrices=False); opts=[]
  for a in np.geomspace(1e-6,1e3,25):
   h=(v*v/(v*v+a)); g=(l*h)@l.T;Y=Z[:52]-Z[:52].mean(0);cv=np.mean(abs((g@Y-Y)/(1-1/52-np.diag(g)[:,None])),0);pred=(f@r.T*(v/(v*v+a)))@l.T@Y+Z[:52].mean(0);err=np.mean(abs(pred[52:]-Z[52:]),0);opts.append((cv,err))
  ids=np.argmin([a[0] for a in opts],axis=0);print(p.shape[1],i,'LOO',[opts[j][0][k] for k,j in enumerate(ids)],'val',[opts[j][1][k] for k,j in enumerate(ids)])
