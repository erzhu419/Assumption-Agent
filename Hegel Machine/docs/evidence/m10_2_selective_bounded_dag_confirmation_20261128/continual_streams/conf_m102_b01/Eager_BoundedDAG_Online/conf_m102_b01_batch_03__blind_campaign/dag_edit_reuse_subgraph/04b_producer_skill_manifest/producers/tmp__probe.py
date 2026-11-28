import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; n,p=u.shape
np.set_printoptions(precision=4,suppress=True,linewidth=138)
print('A samples [u | v]\n',np.c_[u,v][np.argsort(u[:,0])][::2]); print('study ranges',[(k,np.min(s[k]['x'],axis=0),np.max(s[k]['x'],axis=0)) for k in ('train','test')]);print('v corr',np.corrcoef(v.T))
for deg in range(1,5):
 terms=[t for d in range(deg+1) for t in itertools.combinations_with_replacement(range(p),d)]; X=np.array([np.prod(u[:,t],axis=1) for t in terms]).T; B=np.linalg.lstsq(X,v,rcond=None)[0]; R=X@B-v; h=np.sum(np.linalg.svd(X,full_matrices=False)[0]**2,axis=1);print('poly',deg,len(terms),'rmse',np.sqrt(np.mean(R**2,axis=0)),'loo',np.sqrt(np.mean((R/np.maximum(1-h[:,None],1e-10))**2,axis=0))); 
 if deg==3:print(sorted(zip(terms,np.linalg.norm(B,axis=1),B),key=lambda t:-t[1])[:16])
F={};
for i in range(p):
 for j in range(i,p):
  for a in ([0] if i==j else [1,-1,.5,-.5,2,-2]):
   z=u[:,i]+a*u[:,j]; name=(i,j,a)
   for d in range(1,6):F[name+('pow',d)]=z**d
   for a0 in [.5,1,2,3,4,5,6,8,10]:
    for f in [np.sin,np.cos,np.tanh,np.sinh,np.arcsinh,np.arctan]:F[name+(f.__name__,a0)]=f(a0*z)
    F[name+('rational',a0)]=z/(1+a0*z*z)
keys=list(F); X=np.column_stack(list(F.values())); X-=X.mean(0); ns=np.linalg.norm(X,axis=0); X/=np.maximum(ns,1e-12); R=v-v.mean(0)
for j in range(p):
 c=X.T@R[:,j]/np.linalg.norm(R[:,j]); ids=np.argsort(-abs(c))[:12]; print('cor',j,[(keys[i],round(c[i],4)) for i in ids])
