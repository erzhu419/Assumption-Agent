import pickle, itertools, numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; U,V=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(V),axis=1)<3; u,v=U[keep],V[keep]; n,d=u.shape
A=np.c_[np.ones(n),u]; q=np.linalg.qr(A)[0]; residual=lambda x:x-q@(q.T@x); R=residual(v); print('affine coef',np.round(np.linalg.lstsq(A,v,rcond=None)[0],4),'rms',np.sqrt(np.mean(R**2,axis=0)))
cs=[]; names=[]
def add(nm,x):
 cs.append(x); names.append(nm)
for deg in range(2,6):
 for k in itertools.combinations_with_replacement(range(d),deg): add(('monomial',k),np.prod(u[:,k],axis=1))
for i in range(d):
 for k in np.arange(.25,12.01,.05):
  for f in [np.sin,np.cos,np.tanh,np.arcsinh]: add((f.__name__,i,round(k,3)),f(k*u[:,i]))
for i,j in itertools.combinations(range(d),2):
 for sign in [-1,1]:
  for k in np.arange(.25,10.01,.10):
   for f in [np.sin,np.cos,np.tanh]: add((f.__name__,i,j,sign,round(k,3)),f(k*(u[:,i]+sign*u[:,j])))
X=np.array(cs).T; D=residual(X); D/=np.maximum(np.linalg.norm(D,axis=0),1e-13); C=D.T@R
for j in range(d):
 order=np.argsort(-abs(C[:,j])); tops=[]; seen=set()
 for h in order:
  key=names[h][:-1] if names[h][0]!='monomial' else names[h];
  if key in seen: continue
  seen.add(key); tops.append((names[h],round(abs(C[h,j])/np.linalg.norm(R[:,j]),4)))
  if len(tops)==8:break
 print('coordinate',j,'top partial correlations',tops)
 B=A.copy()
 for k in range(5):
  Q=np.linalg.qr(B)[0]; e=v[:,j]-Q@(Q.T@v[:,j]); Z=X-Q@(Q.T@X); norm=np.linalg.norm(Z,axis=0); scores=abs(Z.T@e)/np.maximum(norm,1e-12); scores[norm<1e-8]=0; h=np.argmax(scores); B=np.c_[B,X[:,h]]; err=v[:,j]-B@np.linalg.lstsq(B,v[:,j],rcond=None)[0]; print('  OMP',names[h], np.sqrt(np.mean(err**2)))
for order in range(2,6):
 B=np.column_stack([A]+[u**k for k in range(2,order+1)]); Q=np.linalg.qr(B)[0]; e=(v-Q@(Q.T@v))/(1-np.sum(Q**2,axis=1)[:,None]); print('additive polynomial LOO',order,np.sqrt(np.mean(e**2,axis=0)))
b=s['panel_b']; B=np.c_[np.ones(len(b['u'])),b['u']]; coef=np.linalg.lstsq(B,b['v'],rcond=None)[0]; print('panel B RMSE',np.sqrt(np.mean((B@coef-b['v'])**2)))
