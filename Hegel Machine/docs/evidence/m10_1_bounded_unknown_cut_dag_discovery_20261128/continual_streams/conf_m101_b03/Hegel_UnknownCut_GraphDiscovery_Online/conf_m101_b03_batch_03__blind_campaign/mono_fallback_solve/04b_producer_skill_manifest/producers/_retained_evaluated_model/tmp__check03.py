import pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; z,b,c,d=m._features(s,tr['x']); one=np.ones(len(z)); st=np.column_stack
B=st((one,d,c)); G={'zc':m._outer(z,c),'bc':m._outer(b,c),'bd':m._outer(b,d),'cd':m._outer(c,d),'zd':m._outer(z,d),'bdiag':b*c,'zdiag':z*c,'cdiag':c*d,'cddot':(c*d).sum(1),'zddot':(z*d).sum(1),'bcdot':(b*c).sum(1),'zcdot':(z*c).sum(1),'brat':b/(1+b*b),'znorm':z/(1+(z*z).sum(1,keepdims=True))}
models=['zc cddot','zc cddot zddot','zc cddot zddot cd','zc cddot zddot cdiag','zc cddot zddot cd znorm','zc cd','zc cd zd','zc cd bd','bdiag zdiag cddot','bdiag zdiag cddot zddot','bdiag zdiag cd zddot','bdiag cddot','bcdot zcdot cddot','bdiag cd zddot']
splits=[np.arange(72),np.random.default_rng(19).permutation(72),np.random.default_rng(47).permutation(72),np.argsort(np.max(abs(z),axis=1))];
for keys in models:
 D=st((B,*[G[k] for k in keys.split()])); scores=[]
 for split in splits:
  a,e=split[:52],split[52:]; C=m._fit(D[a],Y[a]); E=abs(D[e]@C-Y[e]);scores.append([E[:,0].mean(),E[:,1:].mean(),E[:,-1].mean()]);
 print(keys,np.round(scores,5).tolist(), 'train',np.round(abs(D[a]@C-Y[a]).mean(0)[[0,8]],5))
for keys in ['zc cddot','bdiag zdiag cddot','zc cd zddot']:
 D=st((B,*[G[k] for k in keys.split()])); C=m._fit(D[:52],Y[:52]); print('coef',keys,np.round(C[:,[0,1,4,8]].T,3));
for idx in (52,56):
 stest=dict(s,train={**tr,**{k:tr[k][:idx] for k in ('x','y','traj')}},test={'x':tr['x'][idx:]}); p=m.solve_batch(stest); print('solver',idx,abs(p['y_hat']-tr['y'][idx:]).mean(),abs(p['state_pred']-tr['traj'][idx:]).mean(),abs(p['state_pred']-tr['traj'][idx:]).mean(0));
