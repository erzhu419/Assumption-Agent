exec(open('/tmp/compare.py').read().split('for design,label')[0]);from scipy.linalg import solve
te=m._decode(s['test']['x'],A); print('ranges',z.min(0),z.max(0),te.min(0),te.max(0)); lam=.001
pairs=list(combinations(range(4),2))
def kernels(v,scale,l,kind):
 d=(v[:,None,:]-v[None,:,:])/scale/l
 k=np.exp(-.5*d*d) if kind=='rbf' else (lambda q:(1+q+q*q/3)*np.exp(-q))(np.sqrt(5)*abs(d))
 return k,np.stack([k[:,:,p]*k[:,:,q] for p,q in pairs],axis=2)
settings=[(l,lam,kind,mix,pw) for l in [.65,.85,1.05,1.25,1.5,1.8,2.2] for lam in [.0003,.001,.003,.01] for kind in ['rbf','mat'] for mix in [0,.5,1.] for pw in [.35,1.,3.]]
def train_kernel(i,j,setting):
 l,lam,kind,mix,pw=setting; sd=z[i].std(0); v=(z-z[i].mean(0))/sd; k,p=kernels(z,sd,l,kind); Kg=k.mean(2)+pw*p.mean(2); H=m._scalar_design(z,ops);H=(H-H[i].mean(0))/H[i].std(0); K=Kg+1+v@v.T+mix*(H@H.T)/H.shape[1]; alpha=solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),y[i],assume_a='pos');return K[np.ix_(j,i)]@alpha
errs=np.array([[np.mean(abs(train_kernel(i,j,q)-y[j])) for i,j in folds] for q in settings]);
print('top mean',*[(settings[k],errs[k,0],errs[k,1:].mean(),errs[k].max()) for k in np.argsort(errs[:,1:].mean(1))[:16]],sep='\n')
print('top worst',*[(settings[k],errs[k,0],errs[k,1:].mean(),errs[k].max()) for k in np.argsort(errs.max(1))[:12]],sep='\n')
for q in [(1.1,.001,'rbf',0.,1.),(1.1,.001,'rbf',.5,1.)]:
 i,j=folds[0];sd=z[i].std(0);v=(z-z[i].mean(0))/sd;k,p=kernels(z,sd,q[0],q[2]);K=1+v@v.T+k.mean(2)+p.mean(2);alpha=solve(K[np.ix_(i,i)]+q[1]*np.eye(len(i)),y[i],assume_a='pos');print('anova',[(str(g),round(np.std(Kg[:,i]@alpha),5)) for g,Kg in list(enumerate(k.transpose(2,0,1)/4))+list(zip(pairs,p.transpose(2,0,1)/6))]);print('err',np.round(train_kernel(i,j,q)-y[j],4))
for label,F in [('z affine',D(z)),('w no offset',w),('w affine',D(w))] if 'D' in globals() else [('z affine',m._affine(z)),('w no offset',w),('w affine',m._affine(w))]:
 cs=np.linalg.lstsq(F,t,rcond=None)[0]; ut,st,vt=np.linalg.svd(cs,full_matrices=False);print(label,'coeff svd',st,'res',np.mean(abs(F@cs-t)))
 for deg in [0,1,2,3,4,5]:
  T=s['eval_times'];S=np.c_[T,1-np.exp(-T),1-np.exp(-2*T),1-np.exp(-3*T)] if deg==0 else np.c_[*[T**k for k in range(1,deg+1)]];e=[]
  for i,j in folds:
   C=m._fit(F[i],t[i]);pred=F[j]@C@S@np.linalg.pinv(S);e.append(np.mean(abs(pred-t[j]),axis=0))
  print(deg,np.mean(e[0]),np.mean(e),np.max(e))
