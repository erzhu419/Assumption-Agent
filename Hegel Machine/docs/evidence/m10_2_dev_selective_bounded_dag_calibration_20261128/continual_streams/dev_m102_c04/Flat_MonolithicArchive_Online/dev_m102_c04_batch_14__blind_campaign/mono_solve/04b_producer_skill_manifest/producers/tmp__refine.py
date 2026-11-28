from identify import *
import runpy
m=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_11_solver.py')
z=m['_calibrate'](s['panel_a'],s['train']['x']); c=s['panel_c']; u=c['u']; v=c['v1']; qf=lambda x:D(funcs['powrat'](x,.85),off(x)[:,[0,2,3,5]]); q=qf(z)@ls(qf(u),c['v2']); t=s['eval_times']; y=s['train']['y']; tr=s['train']['traj']; tb=lambda k:-np.expm1(-np.outer(t,k)); rates=least_squares(lambda k:(tr[:52]-tr[:52]@np.linalg.pinv(tb(k)).T@tb(k).T).ravel(),[1.8,.42]).x; a=tr@np.linalg.pinv(tb(rates)).T; T=np.column_stack((y,a)); result=[]
ring=lambda x:off(x)[:,[0,2,3,5]]
tri=lambda x:np.stack([np.prod(x[:,j],axis=1) for j in it.combinations(range(4),3)],axis=1)
extra={'none':lambda x:x[:,:0],'pair':off,'ring':ring,'tri':tri,'sq':lambda x:x*x,'qf':lambda x:qf(x)[:,1:],'weighted':lambda x:x*(np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)),'opp':lambda x:off(x)[:,[1,4]],'cubicring':lambda x:x*(np.roll(x,1,axis=1)**2+np.roll(x,-1,axis=1)**2),'cubopp':lambda x:x*np.roll(x,2,axis=1)**2}
for name,f in funcs.items():
 for en,g in extra.items():
  for linear in (False,True):
   def feat(x,k):return D(x if linear else x[:,:0],f(x,k),g(x))
   def fit(k):
    X=feat(u,k); w=ls(X,v); e=X@w-v
    return X,w,e
   opt=minimize_scalar(lambda k:np.linalg.norm(fit(k)[2]),bounds=(.015,3.2),method='bounded'); X,w,e=fit(opt.x); loo=np.linalg.norm(e/np.maximum(1e-5,1-np.sum(X*np.linalg.pinv(X).T,axis=1))[:,None])/np.sqrt(v.size); p=feat(z,opt.x)@w; K=(p[:,:,None]*q[:,None,:]).reshape(-1,16); base=D(p,q,np.sum(p*q,axis=1)); broad=D(p,q,K); pred=broad@ls(broad[:52],T[:52]); cv=np.mean(abs(pred[52:,0]-y[52:])); st=np.mean(abs(pred[52:,1:]@tb(rates).T-tr[52:])); result.append((loo,np.std(e),name,en,linear,opt.x,cv,st))
print('panel rankings');print(*sorted(result)[:15],sep='\n');print('downstream rankings');print(*sorted(result,key=lambda r:r[-2]+r[-1])[:16],sep='\n')
for g in [lambda x:D(x,funcs['tanh'](x,1.48884)),lambda x:D(x,funcs['rat'](x,.849778),off(x))]:
 p=g(z)@ls(g(u),v); K=(p[:,:,None]*q[:,None,:]).reshape(-1,16); base=D(p,q,np.sum(p*q,axis=1));
 for col in range(3):
  best=[]
  for j in range(16):
   for typ,A in [('full',base),('q',D(q)),('p',D(p)),('pdot',D(p,np.sum(p*q,axis=1)))]:
    F=np.column_stack((A,K[:,j])); W=ls(F[:52],T[:52,col]); erro=F@W-T[:,col]; best.append((np.mean(erro[:52]**2)**.5,np.mean(abs(erro[52:])),typ,j,round(W[-1],3)))
  print('sparse',col,sorted(best)[:6])
