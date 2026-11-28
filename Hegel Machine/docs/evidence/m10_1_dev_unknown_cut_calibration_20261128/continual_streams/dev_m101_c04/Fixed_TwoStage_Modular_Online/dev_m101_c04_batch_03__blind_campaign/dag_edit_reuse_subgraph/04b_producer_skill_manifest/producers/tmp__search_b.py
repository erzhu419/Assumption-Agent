import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for pn,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[pn]; x,y=p['u'],p[key]; print('\n',pn,key)
 base=np.column_stack([np.ones(len(x)),x]); q=np.linalg.qr(base)[0]; R=lambda a:a-q@(q.T@a)
 e=R(y); F={}
 for i,j in itertools.combinations(range(4),2):
  a,b=x[:,i],x[:,j]
  F[f'{i}*{j}']=a*b
  for c in [.5,1,1.5,2,3]:
   for k,z in [('+',a+b),('-',a-b),('p',a*b)]:
    F[f'sin{c}({i}{k}{j})']=np.sin(c*z);F[f'cos{c}({i}{k}{j})']=np.cos(c*z)
  F[f'{i}abs{j}']=a*abs(b);F[f'{j}abs{i}']=b*abs(a)
 for i,a in enumerate(x.T):
  for j in range(2,6):F[f'{i}^{j}']=a**j
  F[f'abs{i}']=abs(a);F[f'{i}abs{i}']=a*abs(a)
  for c in [.5,1,1.5,2,2.5,3,4,5,6]:
   F[f'sin{c}({i})']=np.sin(c*a);F[f'cos{c}({i})']=np.cos(c*a);F[f'tanh{c}({i})']=np.tanh(c*a)
 for ids in itertools.combinations_with_replacement(range(4),3): F[str(ids)]=np.prod(x[:,ids],axis=1)
 names=list(F); ar=R(np.array(list(F.values())).T); ar/=np.maximum(np.linalg.norm(ar,axis=0),1e-12)
 for k in [None,0,1,2,3]:
  z=e if k is None else e[:,[k]]; r=z.copy(); ids=[]
  for t in range(8):
   score=np.linalg.norm(ar.T@r,axis=1);score[ids]=0;ids.append(np.argmax(score)); d=ar[:,ids];r=z-d@np.linalg.lstsq(d,z,rcond=None)[0];h=np.sum(np.linalg.qr(d)[0]**2,axis=1)+np.sum(q*q,axis=1)
   print(k,t+1,names[ids[-1]],round(np.linalg.norm(r)/np.linalg.norm(z),5),round(np.max(abs(r/(1-h)[:,None])),5))
