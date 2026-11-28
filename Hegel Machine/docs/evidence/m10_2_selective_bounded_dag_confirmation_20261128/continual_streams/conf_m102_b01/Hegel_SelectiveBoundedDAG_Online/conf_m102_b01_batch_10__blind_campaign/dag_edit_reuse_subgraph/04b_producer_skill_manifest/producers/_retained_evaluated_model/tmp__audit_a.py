exec(open('/tmp/probe.py').read().split('opts={')[0])
p=S['panel_a']; a,r=p['u'],p['v']; print('shapes',[(k,{j:getattr(w,'shape',None) for j,w in q.items()}) for k,q in S.items() if isinstance(q,dict)]);print('B range',u.min(0),u.max(0));print('X range',X.min(0),X.max(0));
for j in range(4):
 t=a[:,j];s=a[:,(j+1)%4]; f=np.c_[np.ones(len(t)),t,s,t*s,t**3];y=r[:,j];coef=ls(f,y)
 for _ in range(8):
  ix=np.argsort(abs(f@coef-y))[:int(.875*len(t))];coef=ls(f[ix],y[ix]);
 er=abs(f@coef-y);good=er<max(.001,12*np.median(er));coef=ls(f[good],y[good]); print('A',j,coef,'n_good',sum(good), 'sorted error',np.round(np.sort(abs(f@coef-y)),5))
print('raw and inverted head',cv(X,5),cv(z,5)); print('target quantiles',np.quantile(T['y'],[0,.05,.5,.95,1]),np.quantile(T['traj'][:,-1],[0,.05,.5,.95,1]));print('all keys',list(S))
