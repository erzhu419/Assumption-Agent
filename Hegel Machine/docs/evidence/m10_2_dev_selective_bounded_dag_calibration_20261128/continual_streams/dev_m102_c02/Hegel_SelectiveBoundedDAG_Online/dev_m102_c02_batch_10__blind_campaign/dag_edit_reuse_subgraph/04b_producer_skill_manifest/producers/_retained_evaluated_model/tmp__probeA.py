exec(open('/tmp/fitmaps.py').read().split("tr=s['train']")[0])
from itertools import combinations_with_replacement
u,v=[s['panel_a'][k] for k in ('u','v')]; tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']
def cv(g, split=52):
 st=dict(s,train=dict(x=x[:split],y=y[:split],traj=t[:split])); o=sink.run_node({'__input__':st,'node_00':dict(c=g[:split],query=g[split:])}); return np.mean(abs(o['y_hat']-y[split:])),np.mean(abs(o['state_pred']-t[split:]))
ix2=list(combinations_with_replacement(range(4),2)); ix3=list(combinations_with_replacement(range(4),3)); qr=lambda q,ix:np.column_stack([np.prod(q[:,i],axis=1) for i in ix]); names=['1']+list('abcd')+[''.join('abcd'[j] for j in i) for i in ix2]+[a*3 for a in 'abcd'];
forms=[('pair',lambda q:np.c_[A(q),q*np.roll(q,-1,axis=1),q**3]),('quadratic',lambda q:np.c_[A(q),qr(q,ix2),q**3]),('cubic',lambda q:np.c_[A(q),qr(q,ix2),qr(q,ix3)])]
for n,D in forms:
 d=D(u); idx=np.arange(len(d));
 for _ in range(3):
  r=np.linalg.norm(v-d@ls(d[idx],v[idx]),axis=1); idx=np.argsort(r)[:len(d)-3]
 b=ls(d[idx],v[idx]); print(n,'bad',np.argsort(r)[-5:],'error',errs(d[idx],v[idx]),'max',abs(v[idx]-d[idx]@b).max());
 if n=='quadratic':
  for nm,row in zip(names,b):print(nm,np.round(row,5))
 def f(w):return D(w.reshape(1,4))[0]@b
 rr=[root(lambda w:f(w)-p,p) for p in x]; z=np.array([q.x for q in rr]); print('inversion max',max(np.linalg.norm(q.fun) for q in rr),'cv arch B',cv(old._op(s['panel_b'],'v',z,'b')),'cv raw',cv(z))
 np.save('/tmp/z_'+n,z)
print('B residual singular values')
u,v=[s['panel_b'][k] for k in ('u','v')]; d=A(u); print(np.linalg.svd(v-d@ls(d,v),compute_uv=False)); print('A residual from archived local model vs global maps above')
