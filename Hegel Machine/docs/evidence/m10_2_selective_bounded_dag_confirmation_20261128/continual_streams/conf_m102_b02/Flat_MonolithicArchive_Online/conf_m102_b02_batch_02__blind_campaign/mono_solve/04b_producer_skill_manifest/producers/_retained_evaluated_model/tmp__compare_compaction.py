import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train']; X=np.vstack((tr['x'],s['test']['x']));F,G=m._features(s,X);H=np.c_[F[:,:-4],F[:,-4:].sum(1)]; a=m._calibration(s['panel_a']); z=np.linalg.solve(a[1:].T,(X-a[0]).T).T;b=s['panel_b'];w=m._with_intercept(z)@m._ls(m._with_intercept(b['u']),b['v']);from itertools import combinations_with_replacement as cwr
P=np.c_[m._with_intercept(w),G[:,5:9],np.column_stack([np.prod(w[:,j],axis=1) for j in cwr(range(4),3) if j[0]!=j[2] and not (0 in j and 2 in j and len(set(j))==2)])]; y,T=tr['y'],tr['traj']; ids=np.arange(len(y)); n=len(ids);rng=np.random.default_rng(492)
folds=[(ids[:52],ids[52:])];
for _ in range(200):
 k=rng.permutation(ids);folds.append((k[:52],k[52:]))
for v in [np.linalg.norm(X[:n]-X[:n].mean(0),axis=1),np.sum(np.linalg.qr(G[:n])[0]**2,axis=1),*(sign*X[:n,i] for i in range(4) for sign in (-1,1))]:
 k=np.argsort(v);folds.append((k[:52],k[52:]))
for name,D in [('response',F),('shared interaction response',H),('trajectory',G),('sparse cubic trajectory',P)]:
 E=[]; dest='response' in name
 for a,b in folds:
  if dest: e=np.abs(D[b]@m._ls(D[a],y[a])-y[b]); E.append([e.mean(),e.max()]);continue
  V=np.linalg.svd(T[a],full_matrices=False)[2][:2];e=np.abs(D[b]@m._ls(D[a],T[a]@V.T)@V-T[b]); E.append([e.mean(),e.mean(0).max(),e.max()])
 lev=np.sum((D[n:]@np.linalg.pinv(D[:n]))**2,axis=1)
 print(name,'rank',np.linalg.matrix_rank(D),'fixed',E[0],'random worst',np.max(E[1:201],axis=0),'extreme worst',np.max(E[201:],axis=0),'test leverage max',lev.max())
c=s['panel_c']; u=c['u']; v=c['v1']; i,j=np.array(list(cwr(range(4),2))).T;p=u[:,i]*u[:,j]; E=[]
for k in range(len(u)):
 keep=np.arange(len(u))!=k; ca=m._ls(m._with_intercept(p[keep]),v[keep]);use=np.linalg.norm(ca[1:],axis=1)>.001*np.linalg.norm(ca[1:],axis=1).max();d=m._with_intercept(p[:,use]);l=m._with_intercept(u);E.append([np.abs(l[k]@m._ls(l[keep],v[keep])-v[k]),np.abs(d[k]@m._ls(d[keep],v[keep])-v[k])]);
print('panel C1 LOO MAE, affine vs sparse even:',np.mean(E,axis=0))
