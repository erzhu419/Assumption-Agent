import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; f=m._design(s,x); fit=m._fit
O=f[:,0]; C=f[:,5:9]; E=f[:,9:13]; cross=f[:,13:]; diag=np.array([0,5,10,15]); off=np.setdiff1d(np.arange(16),diag); D=cross[:,diag].sum(1); Fy=np.c_[O,C,E,D]; lv=np.sum(np.linalg.qr(f)[0]**2,axis=1); rng=np.random.default_rng(447)
splits=[('fixed',np.arange(len(x))),('leverage',np.argsort(lv)),('norm',np.argsort(np.linalg.norm(x,axis=1)))]+[(f'x{j}/{sg}',np.argsort(sg*x[:,j])) for j in range(4) for sg in [-1,1]]+[('random',rng.permutation(len(x))) for _ in range(100)]
rows=[]; supports=[]
for name,ids in splits:
 a,b=ids[:52],ids[52:]; q=m._basis(t,T[a]); amp=fit(q,T[a].T).T; full=fit(f[a],amp); p0=f[b]@full@q.T
 # Select the interaction using only spatial coefficients from the training trajectories.
 c=fit(f[a],T[a]); j=off[np.argmax(np.linalg.norm(c[13+off],axis=1))]; supports.append(j)
 G=np.c_[Fy,cross[:,j]]; L=np.c_[G,f[:,1:5]]; U=np.c_[O,C,E,cross[:,diag],cross[:,j]]
 y0=f[b]@fit(f[a],y[a]); y1=Fy[b]@fit(Fy[a],y[a]); y2=G[b]@fit(G[a],y[a]);
 def err(yy,tt):
  e=np.abs(tt-T[b]); return [np.abs(yy-y[b]).mean(),e.mean(),e.mean(0).max()]
 R=[err(y0,p0),err(y1,p0),err(y2,p0)]; Q=q@np.linalg.pinv(q)
 for H in (G,L,U):
  P=H[b]@fit(H[a],T[a]); R.append(err(y1,P@Q)); R.append(err(y1,P))
 rows.append(R)
 if name in ('fixed','leverage','norm'): print(name,np.round(R,6).tolist())
print('worst:',np.round(np.max(rows,axis=0),6).tolist()); print('mean:',np.round(np.mean(rows,axis=0),6).tolist()); print('supports:',np.unique(supports,return_counts=True))
