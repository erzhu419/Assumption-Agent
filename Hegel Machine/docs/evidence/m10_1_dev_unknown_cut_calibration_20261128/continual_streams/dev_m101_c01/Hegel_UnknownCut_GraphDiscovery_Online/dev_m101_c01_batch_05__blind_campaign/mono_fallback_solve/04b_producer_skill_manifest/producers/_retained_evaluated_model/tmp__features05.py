exec(open('/tmp/probe05.py').read().split('p=temp(z[:56])')[0])
def poly(q,k): return np.column_stack([np.ones(len(q))]+[np.prod(q[:,ii],axis=1) for j in range(1,k+1) for ii in combinations_with_replacement(range(q.shape[1]),j)])
for name in ('v1','v2'):
 u=s['panel_c']['u']; target=s['panel_c'][name]; P=poly(u,2)
 print('C panel',name,'quad rms',np.std(P@np.linalg.lstsq(P,target,rcond=None)[0]-target))
 for k in range(1,5):
  uv=np.concatenate([u,np.sin(k*u),np.tanh(k*u)],axis=1); f=np.c_[P,uv[:,4:]]; coef=np.linalg.lstsq(f,target,rcond=None)[0]; h=np.linalg.svd(f,full_matrices=False)[0]; err=(f@coef-target)/np.maximum(1e-9,1-np.sum(h*h,axis=1))[:,None]; print(k, np.mean(abs(err)))
q=a(x); coords={'A':q,'BA':b(q),'B':b(x),'AB':a(b(x)),'C':c(q),'D':d(q)}
rng=np.random.default_rng(47); folds=[(np.arange(56),np.arange(56,72))]+[(r[:56],r[56:]) for r in (rng.permutation(72) for _ in range(5))]
for label in ['direct','amp-assisted']:
 out={}
 for i,j in folds:
  p=temp(z[i]); v=amp(z,p); Y=np.c_[y,v]; F=poly(q,2)
  if label=='amp-assisted': F=np.c_[F,v]
  def run(key,f):
   f=f.copy(); f[:,1:]/=np.maximum(f[i,1:].std(axis=0),1e-8); A=f[i]; beta=np.linalg.solve(A.T@A+1e-4*np.eye(f.shape[1]),A.T@Y[i]); out.setdefault(key,[]).append(abs(f[j]@beta-Y[j]).mean(0))
  run('quadratic',F)
  for name,w in coords.items():
   for k in (.5,1,1.5,2):
    for fn in (np.sin,np.tanh,np.cos):
     g=fn(k*w); run((name,fn.__name__,k),np.c_[F,g,w*g]); run((name,fn.__name__,k,'cross'),np.c_[F,g,*(g*w[:,h:h+1] for h in range(4))])
  if label=='amp-assisted':
   run('amp^2',np.c_[F,*(v[:,h:h+1]*v[:,h:] for h in range(2))]); run('amp-x',np.c_[F,*(v[:,h:h+1]*q for h in range(2))]); run('amp-x^2',np.c_[F,*(v[:,h:h+1]*q for h in range(2)),*(v[:,h:h+1]*v[:,h:] for h in range(2))])
 print(label)
 for u in range(3 if label=='direct' else 1):
  for k,e in sorted(out.items(),key=lambda ke:np.mean(np.array(ke[1])[:,u]))[:8]:
   e=np.array(e)[:,u];print(u,k,'first/mean/worst',e[0],e.mean(),e.max())
