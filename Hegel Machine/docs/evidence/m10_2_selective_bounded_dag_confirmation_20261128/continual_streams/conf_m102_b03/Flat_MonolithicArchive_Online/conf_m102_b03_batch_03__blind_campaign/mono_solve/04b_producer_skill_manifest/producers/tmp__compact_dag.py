import importlib.util,pickle,numpy as np
p=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; a=s['panel_a']; u,v=a['u'],a['v']; U=m._design(u); P=m._ls(U,v); keep=np.where(np.linalg.norm(U@P-v,axis=1)<3)[0]; e=[]
for i in keep:
 j=keep[keep!=i]; P=m._ls(U[j],v[j]); zaff=np.linalg.solve(P[1:].T,v[i]-P[0]); zdiag=np.array([(v[i,k]-p[0])/p[1] for k in range(u.shape[1]) for p in [m._ls(U[j][:,[0,k+1]],v[j,k])]]); zcyc=m._calibrate(dict(u=u[j],v=v[j]),v[i:i+1])[0]; e.append([zdiag-u[i],zaff-u[i],zcyc-u[i]])
print('Held-out reference inverse RMSE (diagonal, full affine, candidate cyclic):',np.sqrt(np.mean(np.square(e),axis=(0,2))))
x=tr['x']; z=m._calibrate(a,x); c,d=m._branches(s['panel_c'],z); M,S=m._features(s,x); pair=(c[:,:,None]*d[:,None,:]).reshape(len(x),16); dot=np.sum(c*d,axis=1); n=len(x); rng=np.random.default_rng(512); order=np.argsort(np.linalg.norm(z,axis=1)); splits=[np.arange(52,n),order[-20:],order[:20]]+[rng.choice(n,20,False) for _ in range(30)];
for j in range(4):
 order=np.argsort(z[:,j]); splits.extend([order[:12],order[-12:]])
fast={'dot+c':m._design(np.c_[c,dot]),'dot+cd':M,'diag+cd':m._design(np.c_[c,d,c*d]),'full':S}; slow={'d':m._design(d),'dc':m._design(np.c_[d,c]),'full':S}; results={}
for iv in splits:
 it=np.setdiff1d(np.arange(n),iv); t=s['eval_times']; Y=tr['traj'][it]; H=m._time_basis(t,Y); amps=m._ls(H,Y.T).T
 def sparse_slow(k):
  F=slow['d']; active=[]
  for _ in range(k):
   X=np.c_[F,pair[:,active]]; R=pair[it]-X[it]@m._ls(X[it],pair[it]); err=amps[:,1]-X[it]@m._ls(X[it],amps[:,1]); score=np.abs(err@R)/np.maximum(np.linalg.norm(R,axis=0),1e-10); score[active]=-1; active.append(np.argmax(score))
  return np.c_[F,pair[:,active]],active
 candidates={**slow,**{f'omp{k}':sparse_slow(k)[0] for k in [1,2,4,6]}}; printfirst=not results
 if printfirst: print('Slow bilinear selected on 52 only:',sparse_slow(6)[1])
 for fn,F in fast.items():
  for gn,G in candidates.items():
   # Joint fit avoids estimating ill-conditioned amplitudes independently when regressors differ.
   def mat(j): return np.column_stack(((F[j,None,:]*H[None,:,0,None]).reshape(len(j)*len(t),-1),(G[j,None,:]*H[None,:,1,None]).reshape(len(j)*len(t),-1)))
   err=np.abs((mat(iv)@m._ls(mat(it),Y.ravel())).reshape(len(iv),len(t))-tr['traj'][iv]); results.setdefault((fn,gn),[]).append([err.mean(),err.mean(0).max(),err[:,-1].mean()])
for k,v in sorted(results.items(),key=lambda kv: np.max(kv[1],axis=0)[1]): print(k,'required',np.round(v[0],6),'worst',np.round(np.max(v,axis=0),6),'mean',np.round(np.mean(v,axis=0),6))
