import pickle,numpy as np
from batch_solver import parts, times, fit, kernel, solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); z,b=parts(s)
from sklearn.model_selection import KFold
splits=[(np.arange(52),np.arange(52,n))]+list(KFold(4,shuffle=True,random_state=921).split(z[:n]))
for key,idx in [('z_norm',np.linalg.norm(z[:n],axis=1)),('b_norm',np.linalg.norm(b[:n],axis=1))]:
 j=np.argsort(idx)[-18:]; splits.append((np.setdiff1d(np.arange(n),j),j))
X=np.c_[.3*z+b,.3*z-b]; W=np.c_[.3*z,b]; E=[]; models=[]
for a in ('sum_asinh','sep_asinh','sum_linear','sep_linear','sum_add','sep_add','blend','operator'):
 for k in (.15,.4,.7,1.4):
  for p in (2,3):
   for rg in (1e-6,1e-4,.001):
    if a=='blend': G=.6*kernel(np.arcsinh(X),p,k)+.4*kernel(np.arcsinh(W),p,k)
    elif a=='operator': G=kernel(np.c_[.3*z,b,.7*np.tanh(1.25*X)],p,k)
    else:
     Q=X if a.startswith('sum') else W
     if a.endswith('asinh'): Q=np.arcsinh(Q)
     G=kernel(Q,p,k)
     if a.endswith('add'):
      H=np.arcsinh(Q)
      G=.4*G+.3*kernel(H,p,k)+.3*sum(kernel(H[:,i:i+1],p,k) for i in range(H.shape[1]))/H.shape[1]
    models.append((a,k,p,rg,G))
TF=[]
for i,j in splits:
 sub=dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':tr['x'][j]}); h=solve_batch(sub); d=abs(h['state_pred']-tr['traj'][j]);print('saved solver',len(i),len(j),'y',np.mean(abs(h['y_hat']-tr['y'][j])),'traj',d.mean(),'endpoint',d[:,-1].mean(),'worst horizon',d.mean(0).max()); f=times(sub); TF.append((f,np.c_[tr['y'],fit(f,tr['traj'].T).T]));
for a,k,p,rg,G in models:
 es=[];preds=[]
 for (i,j),(f,Y) in zip(splits,TF):
  h=G[j][:,i]@np.linalg.solve(G[i][:,i]+rg*np.eye(len(i)),Y[i]); d=h[:,1:]@f.T-tr['traj'][j]; ey=h[:,0]-tr['y'][j];es.append([np.mean(abs(ey)),np.mean(abs(d)),np.mean(abs(d[:,-1]))]);preds.append((ey,d))
 E.append((np.array(es),(a,k,p,rg),preds))
for h in (0,1):
 def score(a):
  e=a[0][:,h];return np.mean(e[1:5])+.3*np.max(e[1:5])+.2*np.mean(e[5:])
 A=sorted([x for x in E if x[0][0,h]<.038],key=score)[:12]
 for e,m,_ in A[:5]: print('head',h,m,np.round(e[:,h],5))
 for i,v in enumerate(A):
  for w in A[:i]:
   for frac in [.25,.5,.75]:
    e=np.array([np.mean(abs(frac*u[h]+(1-frac)*t[h])) for u,t in zip(v[2],w[2])]);
    if e[0]<.038 and e[1:5].max()<A[0][0][1:5,h].max()*.9 and e[5:].mean()<A[0][0][5:,h].mean(): print('blend',h,v[1],w[1],frac,np.round(e,5))
