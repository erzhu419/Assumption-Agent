import sys,pickle,numpy as np,itertools as it
sys.path.insert(0,'/holdout_task'); from node_01 import _coordinates,_fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; Y=np.c_[tr['y'],tr['traj']]; n=len(Y); z,b,c,d,g=_coordinates(s,tr['x']);
q=-np.expm1(-t[:,None]*np.array([.39477,1.8217,4.81594])); A=np.c_[Y[:,0],_fit(q,Y[:,1:].T).T];
p=z*np.roll(z,-1,axis=1); G={'z':z,'p':p,'g':g}; keys=list(G); blocks={k:v for k,v in G.items()};
for i,k in enumerate(keys):
 for j in keys[i:]:
  a,b=G[k],G[j]; blocks[k+j]=np.column_stack([a[:,ii]*b[:,jj] for ii in range(4) for jj in range(4) if k!=j or ii<=jj]);
blocks['fourier']=np.column_stack([f(k*z) for f in (np.sin,np.cos) for k in (1.2,3.6,4.8,6.)]); blocks['higher']=np.column_stack([np.prod(z[:,ix],axis=1) for ix in it.combinations_with_replacement(range(4),3)]);
# All small combinations of blocks, with five independent outer test slices.
rng=np.random.default_rng(72); folds=[np.arange(52,72),*np.array_split(rng.permutation(n),4)]; results=[]
for mask in range(1,1<<len(blocks)):
 ids=[k for i,k in enumerate(blocks) if (mask>>i)&1]; f=np.column_stack([np.ones(n)]+[blocks[k] for k in ids]);f=f/np.maximum(np.linalg.norm(f,axis=0),1e-12); u,sv,_=np.linalg.svd(f,full_matrices=False); u=u[:,sv>1e-8];
 if u.shape[1]>52:continue
 err=[]
 for b in folds:
  a=np.setdiff1d(np.arange(n),b); pred=u[b]@_fit(u[a],A[a]); err.append(np.abs(pred-A[b]).mean(0));
 e=np.array(err); results.append((e.mean(0),e[0],e.max(0),ids,u.shape[1]));
for i in range(4):
 print('OUTPUT',i)
 for e,f,w,ids,dim in sorted(results,key=lambda r:r[0][i])[:7]: print(e.round(4),f.round(4),w.round(4),ids,dim)
# Inspect the odd part independently via parity-structured features, and sparse posterior search over interpretable panel pairs.
Z=np.c_[z,c,d]; names=[k+str(i) for k in 'zcd' for i in range(4)]; atoms=[(a,u) for a,u in zip(names,Z.T)]; atoms.extend([(names[i]+'*'+names[j],Z[:,i]*Z[:,j]) for i,j in it.combinations_with_replacement(range(12),2)]); F=np.column_stack([a for _,a in atoms]); F-=F.mean(0); F/=np.maximum(np.linalg.norm(F,axis=0),1e-9);
for j in range(A.shape[1]):
 a=np.arange(52); bb=np.arange(52,n); h=A[a,j]-A[a,j].mean(); M=F-F[a].mean(0); M/=np.maximum(np.linalg.norm(M[a],axis=0),1e-9); best=[]
 for rnd in range(40):
  ii=[] if rnd==0 else list(rng.choice(F.shape[1],4,replace=False));
  for k in range(len(ii),22):
   er=h-M[a][:,ii]@_fit(M[a][:,ii],h); sc=abs(M[a].T@er); sc[ii]=-1;ii.append(int(np.argmax(sc)));
   for rep in range(5):
    old=ii.copy()
    for i in rng.permutation(len(ii)):
     jj=ii[:i]+ii[i+1:]; v=np.linalg.svd(M[a][:,jj],full_matrices=False)[0][:,:len(jj)]; er=h-v@(v.T@h); sc=abs(M[a].T@er)/np.sqrt(np.maximum(1-np.sum((v.T@M[a])**2,axis=0),1e-8));sc[jj]=-1;ii[i]=int(sc.argmax());
    if old==ii:break
   w=_fit(M[a][:,ii],h); err=np.std(h-M[a][:,ii]@w); te=np.mean(abs(M[bb][:,ii]@w+A[a,j].mean()-A[bb,j])); best.append((te,err,[atoms[i][0] for i in ii]));
 print('SPARSE',j,sorted(best)[:3],flush=True)
