import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
u,v=[s['panel_a'][k] for k in ('u','v')]; d=4
print('A rows: u | v | v-u'); print(np.c_[u,v,v-u])
for p in ('panel_a','panel_b'):
 u,v=[s[p][k] for k in ('u','v')]; n=len(u)
 print(p)
 for js in range(4):
  j=(js+1)%4; k=(js+2)%4; l=(js+3)%4; y=v[:,js]
  fs={'1':np.ones(n)}
  for di in (1,2,3):
   fs.update({''.join(map(str,ii)):np.prod(u[:,ii],axis=1) for ii in itertools.combinations_with_replacement(range(4),di)})
  for h in range(4):
   fs.update({f'{fun}{h}':f(u[:,h]) for fun,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh)]})
  def fit(keys):
   x=np.stack([fs[k] for k in keys],1); keep=np.arange(n); nt=39 if p=='panel_a' else n
   for _ in range(12):
    c=np.linalg.lstsq(x[keep],y[keep],rcond=None)[0]; r=y-x@c; keep=np.argsort(abs(r))[:nt]
   return np.sqrt(np.mean(r[keep]**2)), c, r
  keys=['1',*(str(i) for i in range(4))];
  for stage in range(5):
   if stage:
    keys.append(min((fit(keys+[z])[0],z) for z in fs if z not in keys)[1]);
   err,c,r=fit(keys); print(js,err,keys,np.round(c,4),'largest',np.round(np.sort(abs(r))[-10:],4) if stage in (0,4) else '')
print('train scale',np.quantile(np.abs(s['train']['x']),[.5,.9,.99,1],axis=0))
