import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
u,v=[s['panel_a'][k] for k in ['u','v']]; aff=lambda x:np.c_[np.ones(len(x)),x]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
a=aff(u); e=np.linalg.norm(v-a@ls(a,v),axis=1); keep=e<3*np.median(e); print('outliers',np.where(~keep)[0]); print('affine',ls(a[keep],v[keep])); print('residuals',np.sort(np.linalg.norm((v-a@ls(a[keep],v[keep]))[keep],axis=1)))
for j in range(4):
 x=u[keep]; y=v[keep,j]; terms={'const':np.ones(len(x)),**{f'x{k}':x[:,k] for k in range(4)}}; a=aff(x)
 cand={f'x{k}^{p}':x[:,k]**p for k in range(4) for p in [2,3,4,5]}; cand.update({f'x{k}x{l}':x[:,k]*x[:,l] for k,l in itertools.combinations(range(4),2)}); cand.update({f'x{k}x{l}x{m}':x[:,k]*x[:,l]*x[:,m] for k,l,m in itertools.combinations(range(4),3)})
 for k in range(4):
  for w in np.arange(.5,4.01,.1):
   cand[f'sin({k},{w:.1f})']=np.sin(w*x[:,k]); cand[f'cos({k},{w:.1f})']=np.cos(w*x[:,k]); cand[f'tanh({k},{w:.1f})']=np.tanh(w*x[:,k])
 names=[]
 for i in range(7):
  r=y-a@ls(a,y); q=np.linalg.qr(a)[0]; c=np.column_stack(list(cand.values())); c=c-q@(q.T@c); ix=np.argmax(abs(r@c)/np.maximum(1e-10,np.linalg.norm(c,axis=0))); k=list(cand)[ix]; a=np.c_[a,cand.pop(k)]; names.append(k); print(j,names[-1],np.sqrt(np.mean((y-a@ls(a,y))**2)))
print('panel_c v1 expanded candidates')
u=s['panel_c']['u']; v=s['panel_c']['v1'];
for typ,f in [('tanh',np.tanh),('sqrt',lambda x:x/np.sqrt(1+x*x)),('rat',lambda x:x/(1+x*x)),('abs',lambda x:x/(1+abs(x)))]:
 p={f'x{k}^{n}':u[:,k]**n for k in range(4) for n in range(2,6)}; p.update({f'x{k}x{l}':u[:,k]*u[:,l] for k,l in itertools.combinations(range(4),2)}); p.update({f'x{k}²x{l}':u[:,k]**2*u[:,l] for k in range(4) for l in range(4) if l!=k}); p.update({f'x{k}x{l}x{m}':u[:,k]*u[:,l]*u[:,m] for k,l,m in itertools.combinations(range(4),3)}); p.update({str(i):u[:,i]*f(u[:,i]) for i in range(4)})
 arr=[]
 for name,col in p.items():
  def fun(w):
   a=np.c_[aff(u),f(w*u),col]; q=np.linalg.qr(a)[0]; r=v-q@(q.T@v); return np.sqrt(np.mean((r/(1-(q*q).sum(1))[:,None])**2))
  opt=minimize_scalar(fun,bounds=(.2,3),method='bounded'); arr.append((opt.fun,name,opt.x))
 print(typ,sorted(arr)[:6])
