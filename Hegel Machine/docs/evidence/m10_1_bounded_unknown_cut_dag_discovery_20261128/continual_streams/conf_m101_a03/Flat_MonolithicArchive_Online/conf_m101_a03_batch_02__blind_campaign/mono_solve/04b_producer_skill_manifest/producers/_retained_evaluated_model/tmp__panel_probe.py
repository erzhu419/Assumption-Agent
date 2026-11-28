from measure import *
u,v=[s['panel_a'][k] for k in ('u','v')]; A=np.c_[np.ones(len(u)),u]; ids=np.arange(len(u)); keep=ids
for _ in range(3): keep=np.argsort(np.linalg.norm(v-A@m._fit(A[keep],v[keep]),axis=1))[:45]
print('affine trim',np.sort(np.setdiff1d(ids,keep)))
for d in (2,3,4):
 terms=[inds for j in range(d+1) for inds in combinations_with_replacement(range(4),j)] if False else None
 from itertools import combinations_with_replacement
 terms=[inds for j in range(d+1) for inds in combinations_with_replacement(range(4),j)]
 F=np.column_stack([np.prod(u[:,t],axis=1) for t in terms]); C=m._fit(F[keep],v[keep]); e=v-F@C
 print('degree',d,'rms',np.sqrt(np.mean(e[keep]**2)), 'largest residual rows',[(int(i),round(float(np.linalg.norm(e[i])),5)) for i in np.argsort(np.linalg.norm(e,axis=1))[-7:]])
 if d<=3:
  for i in range(4): print(i,[(t,round(float(c),5)) for t,c in zip(terms,C[:,i]) if abs(c)>.012])
r=np.roll(u,-1,axis=1); D=np.stack([np.ones_like(u),u,r,u*r,r**3],axis=1)
for it in range(4):
 c=np.column_stack([m._fit(D[keep,:,i],v[keep,i]) for i in range(4)])
 e=v-np.einsum('nki,ki->ni',D,c)
 norm=np.linalg.norm(e,axis=1)
 print('cyclic',it,'rms',np.std(e[keep]),'worst retained',[(int(i),round(float(norm[i]),6)) for i in keep[np.argsort(norm[keep])[-7:]]); print(c.T)
 keep=np.argsort(norm)[:45]
