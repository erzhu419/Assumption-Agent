exec(open('/tmp/grammar.py').read().split('rows=[]')[0])
from scipy.linalg import qr
from sklearn.linear_model import lars_path
np.set_printoptions(precision=4,suppress=True)
p=s['panel_b']; print('B residual',np.mean(abs(b(p['u'])-p['v'])), 'telemetry',s['telemetry_vec']); print('train summary',s['train']['summary'])
N=52;Yt=Y[:N];
for name,z in list(trans.items()):
 terms=[j for k in range(4) for j in combinations_with_replacement(range(4),k)]; P=np.column_stack([np.prod(z[:,j],axis=1) for j in terms]); W=fit(P[:N],Yt)
 if name in ('a','a0','a1','ba','ba0','ba1'): print('cubic',name,np.mean(abs(P[:N]@W-Yt),0))
rows=[]
for name in ('a','x','a0','a1','ba','ba0','ba1','Ba','Ba0','Ba1'):
 z=trans[name]; z=(z-z[:N].mean(0))/z[:N].std(0); S=[bias(z)]; groups=[]
 for j in range(4):
  u=z[:,j:j+1]; S.extend([u**2,u**3,u**4,np.tanh(u),np.arctan(u),u/(1+abs(u)),np.exp(-u*u/2)])
 for i,j in combinations(range(4),2):
  u,v=z[:,i:i+1],z[:,j:j+1]; G=[u*v,u*u*v,u*v*v,u*u*v*v,u**3*v,u*v**3,u*v/(1+.5*(u*u+v*v))];
  for sg in [-1,1]:
   w=(u+sg*v)/1.4;G += [np.sin(w),np.cos(w),np.tanh(w),w/(1+abs(w)),np.exp(-w*w/2)]
  groups.append(np.column_stack(G))
 for target in [0,1,2]:
  y=Y[:,target];
  for trang in ['id','asinh','sinh'] if target==0 else ['id']:
   fun,inv={'id':(lambda a:a,lambda a:a),'asinh':(np.arcsinh,np.sinh),'sinh':(np.sinh,np.arcsinh)}[trang]; center=y[:N].mean(); yy=fun(y-center)
   for ngrp in [0,1,2,3,6]:
    scans=[()] if ngrp==0 else list(combinations(range(6),ngrp));
    for g in scans:
     F=np.column_stack(S+[groups[j] for j in g]);F/=np.maximum(np.linalg.norm(F[:N],axis=0),1e-8);K=F@F[:N].T
     vals,vec=np.linalg.eigh(K[:N]); VY=vec.T@yy[:N]
     for reg in [1e-7,1e-5,1e-3,.02]:
      D=1/(vals+reg);w=vec@(D*VY); loo=yy[:N]-w/(vec**2@D); pred=K[N:72]@w
      elo=np.mean(abs(inv(loo)+center-y[:N])); eva=np.mean(abs(inv(pred)+center-y[N:])); rows.append((target,elo,eva,name,trang,g,reg))
   F=np.column_stack(S+groups+[np.prod(z[:,j],axis=1)[:,None] for j in combinations(range(4),3)]); F=F[:,1:];mu=F[:N].mean(0);sd=F[:N].std(0); F=(F-mu)/sd; ym=yy[:N].mean();_,active,C=lars_path(F[:N],yy[:N]-ym,method='lasso',max_iter=120)
   for step in range(4,C.shape[1]):
    ids=np.flatnonzero(abs(C[:,step])>1e-10); G=bias(F[:,ids]); Wi=np.linalg.pinv(G[:N]); w=Wi@yy[:N];h=np.sum(G[:N]*Wi.T,axis=1);loo=yy[:N]-(yy[:N]-G[:N]@w)/np.maximum(1-h,1e-9); pred=G[N:72]@w
    if np.max(abs(loo))>20 or np.max(abs(pred))>20:continue
    rows.append((target,np.mean(abs(inv(loo)+center-y[:N])),np.mean(abs(inv(pred)+center-y[N:])),name,trang,tuple(ids),'lars'))
for target in [0,1,2]:
 r=[z[1:] for z in rows if z[0]==target]; print('target',target,'internal',*sorted(r,key=lambda z:z[0])[:5],sep='\n');print('target',target,'tail',*sorted(r,key=lambda z:z[1])[:4],sep='\n')
