import pickle,numpy as np,runpy,itertools
from scipy.linalg import qr
m=runpy.run_path('/holdout_task/node_00.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=m['_coordinates'](s); M=m['_maps'](s,X); z=X[0][:72]; c,d=(a[:72] for a in M[0]); Z=(z,c,d); Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); rng=np.random.default_rng(123); splits=[np.arange(54,n),np.arange(52,n),*np.array_split(rng.permutation(n),4)];
def poly(x,k):
 return np.column_stack([np.prod(x[:,i],axis=1) for j in range(1,k+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),j)])
R=np.column_stack([z*z/(1+a*z*z) for a in [.12,.25,.45,.8]]); H=np.column_stack([np.tanh(a*z) for a in [.6,1,1.5]]+[np.sin(a*z) for a in [1,2,3]]+[z/(1+a*abs(z)) for a in [.4,1]]); mx=np.c_[z,c,d]; modules={'z2':poly(z,2),'z3':poly(z,3),'z4':poly(z,4),'z5':poly(z,5),'cd':np.c_[c,d],'R':R,'H':H,'zR':(z[:,:,None]*R[:,None,:]).reshape(n,-1),'zH':(z[:,:,None]*H[:,None,:]).reshape(n,-1),'c2':poly(c,2),'d2':poly(d,2),'zc':poly(np.c_[z,c],2),'all2':poly(mx,2)}; out=[]
for tags in ['z2 cd','z3 cd','z4 cd','z5 cd','z2 R','z3 R','z4 R','z2 R H','z3 R H','z4 R H','z3 R zR','z3 R H zH','z2 c2 d2','z2 cd H','zc cd','zc cd R','zc cd R H','all2','all2 R','all2 R H','z3 all2 R H','z4 R H zH zR']:
 a=np.column_stack([modules[t] for t in tags.split()]); folds=[]
 for te in splits:
  tr=np.setdiff1d(np.arange(n),te); mu=a[tr].mean(0); sd=np.maximum(a[tr].std(0),.03); f=(a-mu)/sd; yy=Y[tr]-Y[tr].mean(0); U,S,V=np.linalg.svd(f[tr],full_matrices=False); g=f[te]@V.T; E=[]
  for lam in [1e-9,1e-6,1e-4,.001,.01,.1,1]:
   pred=(g*(S/(S*S+lam)))@(U.T@yy)+Y[tr].mean(0); er=abs(pred-Y[te]); E.append([er[:,0].mean(),er[:,1:].mean()]);
  # Individual greedy sparse regression, with fixed path length for verification.
  for kind in ['omp','forward']:
   es=[]
   F=f[tr]; norm=np.sum(F*F,axis=0).clip(1e-10); ys=yy[:,[0,3,-1]]; cols=[]; arr=[]
   for count in range(24):
    Q=np.linalg.qr(F[:,cols])[0] if cols else F[:,:0]; rem=ys-Q@(Q.T@ys); div=(norm-np.sum((Q.T@F)**2,axis=0)).clip(1e-10) if kind=='forward' else norm; scores=np.sum(((F.T@rem)/np.maximum(ys.std(0),.1))**2,axis=1)/div; scores[cols]=-1; cols.append(int(np.argmax(scores))); pred=f[te][:,cols]@np.linalg.lstsq(F[:,cols],yy,rcond=1e-8)[0]+Y[tr].mean(0); er=abs(pred-Y[te]); arr.append([er[:,0].mean(),er[:,1:].mean()]);
   E.extend(arr[3::4]);
  folds.append(E)
 ff=np.array(folds); train_err=np.linalg.norm(np.c_[np.ones(n),a]@np.linalg.lstsq(np.c_[np.ones(n),a],Y,rcond=1e-8)[0]-Y)/np.sqrt(Y.size); print(tags,'in',round(train_err,5),'best',round(ff.max(axis=(0,2)).min(),4), np.round(ff[0].min(0),4));
 for i in range(ff.shape[1]): out.append((ff[:,i].max(),tags,i,np.round(ff[:,i],4).tolist()));
print('TOP',*sorted(out)[:16],sep='\n')
