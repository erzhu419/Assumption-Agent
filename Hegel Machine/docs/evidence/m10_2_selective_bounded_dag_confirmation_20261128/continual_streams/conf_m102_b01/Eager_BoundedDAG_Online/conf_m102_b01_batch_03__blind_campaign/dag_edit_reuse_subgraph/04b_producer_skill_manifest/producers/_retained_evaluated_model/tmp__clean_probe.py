import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; keep=np.max(abs(v),axis=1)<4; print('Grossly corrupted rows:',np.flatnonzero(~keep));u=u[keep];v=v[keep];p=u.shape[1]; np.set_printoptions(precision=5,suppress=True,linewidth=130)
for deg in range(1,4):
 terms=[t for d in range(deg+1) for t in itertools.combinations_with_replacement(range(p),d)]; X=np.array([np.prod(u[:,t],axis=1) for t in terms]).T; B=np.linalg.lstsq(X,v,rcond=None)[0]; R=X@B-v; h=np.sum(np.linalg.svd(X,full_matrices=False)[0]**2,axis=1);print('poly',deg,'rmse',np.sqrt(np.mean(R**2,axis=0)),'loo',np.sqrt(np.mean((R/(1-h[:,None]))**2,axis=0))); 
 if deg<3: print(list(zip(terms,B)))
X=np.c_[np.ones(len(u)),u]; Q=np.linalg.qr(X)[0]; ortho=lambda x:x-Q@(Q.T@x);R=ortho(v); print('clean residual svd',np.linalg.svd(R,full_matrices=False)[1]); print('max clean v',abs(v).max(0));
D={}
for i in range(p):
 for j in range(i,p):
  for a in ([0] if i==j else np.r_[np.arange(-2,0,.25),np.arange(.25,2.01,.25)]):
   z=u[:,i]+a*u[:,j]
   for d in range(2,5): D[i,j,a,'power',d]=z**d
   for w in np.arange(.5,10.01,.25):
    D[i,j,a,'sin',w]=np.sin(w*z);D[i,j,a,'cos',w]=np.cos(w*z);D[i,j,a,'tanh',w]=np.tanh(w*z);D[i,j,a,'rational',w]=z/(1+w*z*z)
   if i!=j:
    for w in np.arange(.5,10.01,.25):D[i,j,a,'sin_product',w]=np.sin(w*u[:,i]*u[:,j]);D[i,j,a,'cos_product',w]=np.cos(w*u[:,i]*u[:,j]);
keys=list(D); F=ortho(np.column_stack(list(D.values()))); F/=np.maximum(np.linalg.norm(F,axis=0),1e-12); C=F.T@R/np.linalg.norm(R,axis=0)
for j in range(p):
 k=np.argmax(abs(C[:,j])); r=R[:,j]-F[:,k]*(F[:,k]@R[:,j]); c=F.T@r/np.linalg.norm(r); ids=np.argsort(-abs(c))[:4];print('column',j,'first',keys[k],C[k,j],'next',[(keys[i],round(c[i],5)) for i in ids]);
