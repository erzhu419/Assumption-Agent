import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; t=s['eval_times']; T=tr['traj']; y=tr['y']; f=m._design(s,x); fit=m._fit
print('study fields:',list(s),'train fields:',list(tr),'times:',t)
print('calibration offsets/gains:',m._calibration(s['panel_a']))
D=f[:,[13,18,23,28]].sum(1); C=f[:,5:9]; E=f[:,9:13]; z=f[:,1:5]; O=np.ones(len(x)); cross=f[:,13:]; offdiag=np.setdiff1d(np.arange(16),[0,5,10,15]); extras=[]
for j in range(4):
 for k in range(j+1,4):
  for sg in [-1,1]:
   q=z[:,j]+sg*z[:,k]; extras.extend([q**3, np.cos(2.4*q),q/(1+q*q),z[:,j]*z[:,k]/(1+q*q)])
extras=np.column_stack(extras)
leverage=np.sum(np.linalg.qr(f)[0]**2,axis=1); rng=np.random.default_rng(447)
splits=[('fixed',np.arange(len(x))),('leverage',np.argsort(leverage)),('norm',np.argsort(np.linalg.norm(x,axis=1)))]+[(f'x{j}/{sg}',np.argsort(sg*x[:,j])) for j in range(4) for sg in [-1,1]]+[('random',rng.permutation(len(x))) for _ in range(100)]
results=[]; supports=[]
for name,ids in splits:
 a,b=ids[:52],ids[52:]; q=m._basis(t,T[a]); amps=fit(q,T[a].T).T
 full=fit(f[a],np.c_[y[a],amps]); inter=offdiag[np.argmax(abs(full[13+offdiag,1]))]; supports.append(inter)
 Fy=np.column_stack((O,C,E,D)); A=np.column_stack((O,E,cross[:,inter])); B=np.column_stack((O,C,D));
 def time(r):return -np.expm1(-t[:,None]*r)
 def design(a,q):return np.c_[(A[a,:,None]*q[:,0]).transpose(0,2,1).reshape(len(a)*len(t),-1),(B[a,:,None]*q[:,1]).transpose(0,2,1).reshape(len(a)*len(t),-1)]
 def residual(r):
  g=design(a,time(r)); yy=T[a].ravel(); return g@fit(g,yy)-yy
 opt=least_squares(residual,[.42,1.83],bounds=([.12,1.],[.9,4.])); Q=time(opt.x); sy=Fy[b]@fit(Fy[a],y[a]); st=(design(b,Q)@fit(design(a,Q),T[a].ravel())).reshape(len(b),-1)
 base=f[b]@full; Y0=base[:,0]; T0=base[:,1:]@q.T
 # Fit residual augmentations only in the training fold, using mild shrinkage.
 g=extras/np.maximum(.01,np.std(extras[a],axis=0)); Z=np.c_[f,g]; pen=np.diag(np.r_[np.zeros(f.shape[1]),np.full(g.shape[1],.3)]); aug=Z[b]@fit(np.vstack([Z[a],pen]),np.vstack([np.c_[y[a],amps],np.zeros((len(pen),3))]));
 def err(yy,tt):
  e=abs(tt-T[b]); return [abs(yy-y[b]).mean(),e.mean(),e.mean(0).max()]
 results.append(np.array([err(Y0,T0),err(sy,st),err(aug[:,0],aug[:,1:]@q.T)]))
 if name in ('fixed','leverage','norm'): print(name,'inter',inter,'rates',opt.x,'train trajectory MAE',np.mean(abs(opt.fun)),'base/compact/rational-pair',results[-1].tolist())
print('fold maxima:',np.max(results,axis=0)); print('fold means:',np.mean(results,axis=0)); print('fold support counts',np.unique(supports,return_counts=True))
