exec(open('/tmp/scan6.py').read().split('# Screen coordinate')[0])
np.set_printoptions(suppress=True,precision=5,linewidth=150)
print('B coefficients',ls(des(u,erf(1.411208*u)),v)); print('B residual', (v-B(u))[:12]); print('B range',u.min(0),u.max(0),B(ai).min(0),B(ai).max(0)); print('C coeff',ls(des(uc,q(uc)),pc['v1'])); print('trajectory singular values',np.linalg.svd(Y[:,1:],compute_uv=False)); t=s['eval_times']; E=Y[:,1:]; print('trajectory ratios',np.quantile(E[np.abs(E[:,-1])>.5]/E[np.abs(E[:,-1])>.5,-1,None],[0,.25,.5,.75,1],axis=0)); print('y, trajectory corr',np.corrcoef(Y.T)[0]);
# Look for each response in a sparse basis of the candidate fork, including readout nonlinearities.
a0=ai;b0=B(a0);c0=C(b0);d0=D(b0); data={'a':a0,'b':b0,'c':c0,'d':d0,'ca':C(a0),'da':D(a0)}; N=72
blocks={}; labels={};
def add(name,z):
 blocks[name]=z;labels[name]=[name+str(i) for i in range(z.shape[1])]
for name,z in data.items():
 add(name,z);add(name+'q',q(z));add(name+'3',z**3);add(name+'tanh',np.tanh(z));add(name+'sin',np.sin(z));add(name+'asinh',np.arcsinh(z));add(name+'rat',z/(1+z*z));add(name+'abs',abs(z));
for a,b in [('a','b'),('a','c'),('a','d'),('b','c'),('b','d'),('c','d')]:add(a+b,(data[a][:,:,None]*data[b][:,None,:]).reshape(N,-1))
from sklearn.linear_model import LassoLars
splits=[(np.setdiff1d(np.arange(52),te),te) for te in np.array_split(np.random.default_rng(35).permutation(52),4)]+[(np.arange(52),np.arange(52,72))];
G=np.column_stack(list(blocks.values())); names=np.concatenate(list(labels.values())); out=[]
for target in [Y[:,0],Y[:,-1]]:
 print('target',np.std(target)); res=[]
 for a in [.00015,.0005,.0015,.004,.008,.015,.025]:
  errors=[]
  for tr,te in splits:
   mu=G[tr].mean(0);scale=G[tr].std(0);Z=(G-mu)/np.maximum(scale,.000001);reg=LassoLars(alpha=a,max_iter=130).fit(Z[tr],target[tr]);pred=reg.predict(Z[te]);errors.append(np.mean(abs(pred-target[te]))); ws=reg.coef_
  print(a,np.round(errors,5),len(np.flatnonzero(ws)),[(names[i],round(ws[i],3)) for i in np.argsort(-abs(ws))[:12]])
 for n,z in data.items():
  for nm,F in [('lin',z),('quad',np.c_[z,q(z)]),('pure',np.c_[z,np.tanh(z),np.sin(z),np.arcsinh(z)]),('fork',np.c_[a0,z,q(z)]),('fork_diag',np.c_[a0,z,z*z,a0*z]),('fork_full',np.c_[a0,z,q(z),(a0[:,:,None]*z[:,None,:]).reshape(N,-1)])]:
   for reg in [.0001,.001,.01,.1,1]:
    errors=[]
    for tr,te in splits:
     Z=(F-F[tr].mean(0))/np.maximum(.0001,F[tr].std(0)); U,w,V=np.linalg.svd(Z[tr],full_matrices=False);p=target[tr].mean()+(Z[te]@V.T*w/(w*w+reg))@(U.T@(target[tr]-target[tr].mean()));errors.append(np.mean(abs(p-target[te])));
    res.append((np.mean(errors[:4]),errors[-1],n,nm,reg))
 print('top by inner',sorted(res)[:12]);print('top by outer',sorted(res,key=lambda r:r[1])[:8]);
