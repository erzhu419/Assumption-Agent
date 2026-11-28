import numpy as np,pickle,runpy
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s}); b=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']({'node_00':a}); u,z=a['u_tr'],b['z_tr']; y=s['train']['y']; n=len(y); ids=np.arange(n)
x=np.c_[u,z]; nm=[f'{c}{i}' for c in 'uz' for i in range(4)]; L=np.c_[np.ones(n),u]; fs=[]; names=[]
def add(name,v): names.append(name); fs.append(v)
for d in (2,3,4):
 for jj in combinations_with_replacement(range(8),d):
  if d==4 and len(set(jj))>2: continue
  add('*'.join(nm[j] for j in jj),np.prod(x[:,jj],axis=1))
funcs={'sin':np.sin,'tanh':np.tanh,'rat':lambda q:q/(1+q*q),'atan':np.arctan,'soft':lambda q:q/(1+abs(q)),'cos':np.cos,'abs':abs}
for i in range(8):
 for k,f in funcs.items():
  for c in (0.5,1,2):
   F=f(c*x[:,i]); add(f'{k}({c}*{nm[i]})',F)
   for j in range(8): add(f'{nm[j]}*{k}({c}*{nm[i]})',x[:,j]*F)
for i in range(4):
 for j in range(4,8):
  q=x[:,i]*x[:,j]
  for k,f in funcs.items():
   for c in (0.5,1,2): add(f'{k}({c}*{nm[i]}*{nm[j]})',f(c*q))
  for c in (.25,.5,1,2):
   for v,w in ((i,j),(j,i)):
    add(f'{nm[v]}/(1+{c}*{nm[w]}^2)',x[:,v]/(1+c*x[:,w]**2)); add(f'{nm[v]}^2/(1+{c}*{nm[w]}^2)',x[:,v]**2/(1+c*x[:,w]**2))
  for sign in (-1,1):
   q=x[:,i]+sign*x[:,j]
   for k,f in funcs.items():
    for c in (.5,1,2): add(f'{k}({c}*({nm[i]}+{sign}*{nm[j]}))',f(c*q))
X=np.array(fs).T
splits=np.split(np.random.default_rng(982).permutation(n),6)+[np.arange(52,72)]
def greedy(tr,te,steps,base=L, verbose=False):
 d=base.copy(); out=[]
 for t in range(steps+1):
  q=np.linalg.qr(d[tr],mode='reduced')[0]; res=y[tr]-q@(q.T@y[tr]); beta=np.linalg.lstsq(d[tr],y[tr],rcond=None)[0]; F=X[tr]-q@(q.T@X[tr]); norms=np.linalg.norm(F,axis=0); sc=abs(F.T@res)/np.where(norms>1e-7,norms,np.inf); jj=np.argmax(sc)
  out.append(np.mean(abs(d[te]@beta-y[te])))
  if verbose: print(t, 'rmse',np.std(res), 'top', [(names[j],round(sc[j],4)) for j in np.argsort(sc)[-5:][::-1]])
  d=np.c_[d,X[:,jj]]
 return out
Q=np.c_[L,np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)])]
for base in (L,Q):
 print('BASE',base.shape[1]); m=np.array([greedy(np.setdiff1d(ids,te),te,14,base) for te in splits]); print('mean/worst/last',np.round(np.stack([m[:-1].mean(0),m[:-1].max(0),m[-1]]),5)); greedy(ids,ids,8,base,True)
C=s['panel_c']; U,V=s['panel_b'].values(); B=np.linalg.lstsq(np.c_[np.ones(len(U)),U],V,rcond=None)[0]; c=C['u']; cz=np.c_[np.ones(len(c)),c]@B
for k in ('v1','v2'):
 print('Panel C',k,'affine residual',np.std(C[k]-np.c_[np.ones(len(c)),c]@np.linalg.lstsq(np.c_[np.ones(len(c)),c],C[k],rcond=None)[0],axis=0),'v1 proximity to c,cz',np.mean(abs(C[k]-c)),np.mean(abs(C[k]-cz)))
print('telemetry',s['telemetry_vec']); print('train/test u ranges',u.min(0),u.max(0),a['u_te'].min(0),a['u_te'].max(0))
