import itertools as it,pickle,numpy as np
u,v=(lambda b:(b['u'],b['v']))(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['panel_b'])
a=np.c_[np.ones(len(u)),u]; q=np.linalg.qr(a)[0]; p=np.eye(len(u))-q@q.T; r=p@v
print('affine residual singular values',np.linalg.svd(r,compute_uv=False)); print('range',u.min(0),u.max(0)); print('affine coefficients',np.linalg.lstsq(a,v,rcond=None)[0]); print('residual rows',np.round(np.c_[u,r],3))
f={}
for j in range(4):
 x=u[:,j]; f[f'x{j}']=x
 for k in range(j):
  for s in (-1,1): f[f'x{j}{s:+d}x{k}']=x+s*u[:,k]
  f[f'x{j}*x{k}']=x*u[:,k]
d={}
for k in (2,3,4):
 for t in it.combinations_with_replacement(range(4),k): d[str(t)]=u[:,t].prod(1)
for name,x in f.items():
 for scale in (.25,.5,1,1.5,2,3,4):
  for tag,y in [('sin',np.sin(scale*x)),('cos',np.cos(scale*x)),('tanh',np.tanh(scale*x)),('rat',x/(1+scale*x*x)),('rat2',x*x/(1+scale*x*x)),('exp',np.exp(-scale*x*x)),('abs',abs(x)),('soft',np.logaddexp(scale*x,-scale*x))]: d[f'{tag}:{scale}:{name}']=y
names=list(d); F=p@np.column_stack(list(d.values())); F/=np.maximum(np.linalg.norm(F,axis=0),1e-12); S=(F.T@r)**2
for j in range(5):
 score=S[:,j] if j<4 else S.sum(1); seen=set(); best=[]
 for i in np.argsort(score)[::-1]:
  key=names[i].split(':')[-1];
  if key in seen: continue
  seen.add(key); best.append((names[i],round(float(score[i]/np.sum(r[:,j if j<4 else slice(None)]**2)),4)))
  if len(best)==8:break
 print('basis ranking',j,best)
