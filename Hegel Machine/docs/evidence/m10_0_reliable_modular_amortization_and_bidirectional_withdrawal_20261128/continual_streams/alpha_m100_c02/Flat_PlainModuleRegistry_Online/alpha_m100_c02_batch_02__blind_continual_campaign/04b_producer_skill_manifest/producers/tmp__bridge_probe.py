import numpy as np,pickle,itertools
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; print('load design',[(i, np.sum(np.count_nonzero(X,axis=1)==i)) for i in range(5)],'first rows',X[:8]); print('spread',np.quantile(X,[0,.1,.5,.9,1],axis=0));
I=list(range(4)); pairs=list(itertools.combinations(I,2)); P=X.T; n=len(X)
a=np.c_[np.ones(n),X]; q=np.column_stack([P[i]*P[j] for i,j in itertools.combinations_with_replacement(I,2)]); c=np.column_stack([P[i]*P[j]*P[k] for i,j,k in itertools.combinations_with_replacement(I,3)]); w=np.c_[X**2,X**3]; pw=np.column_stack([P[i]*P[j] for i,j in pairs]);
def fit(A,Y):
 b=np.linalg.lstsq(A,Y,rcond=None)[0]
 for _ in range(35):
  r=np.linalg.norm(A@b-Y,axis=1); s=max(.0001,np.median(r)); w=1/(1+(r/(1.6*s))**2); u=w**.5; bb=np.linalg.lstsq(A*u[:,None],Y*u[:,None],rcond=None)[0];
  if np.max(abs(bb-b))<1e-9:break
  b=bb
 return b
b=fit(a,Y); r=np.linalg.norm(a@b-Y,axis=1); print('affine',np.round(b,4)); print('largest residual',[(int(i),round(r[i],3)) for i in np.argsort(r)[-12:]]); clean=r<2
models={'affine':a,'pure quad':np.c_[a,X**2],'full quad':np.c_[a,q],'pure cubic':np.c_[a,w],'quad+pure cubic':np.c_[a,q,X**3],'all cubic':np.c_[a,q,c],'quad+sat':np.c_[a,q,np.tanh(X)-X],'quad+sin':np.c_[a,q,np.sin(X)-X],'pair+sat':np.c_[a,pw,np.tanh(X)-X]}
rng=np.random.default_rng(42); folds=np.array_split(rng.permutation(n),5); axis=np.count_nonzero(X,axis=1)==1; tails=np.max(abs(X),axis=1)>1.2
for name,A in models.items():
 b=fit(A,Y); err=np.linalg.norm(A@b-Y,axis=1); pred=np.empty_like(Y)
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); pred[te]=A[te]@fit(A[tr],Y[tr]);
 e=np.linalg.norm(pred-Y,axis=1); print(name,'in %.5f CV'%np.median(err[clean]),[round(np.sqrt(np.mean(e[m&clean]**2)),5) if np.any(m&clean) else None for m in [np.ones(n,bool),axis,~axis,tails]],'fold worst',round(max(np.sqrt(np.mean(e[t][clean[t]]**2)) for t in folds),5));
 if name in ('full quad','quad+pure cubic','quad+sat'): print('nonlinear coef',np.round(b[5:].T,4))
