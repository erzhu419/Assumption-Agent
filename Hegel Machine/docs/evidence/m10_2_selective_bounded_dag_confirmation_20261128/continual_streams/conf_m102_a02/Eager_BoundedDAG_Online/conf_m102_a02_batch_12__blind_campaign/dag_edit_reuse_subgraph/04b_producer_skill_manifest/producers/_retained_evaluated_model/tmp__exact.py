exec(open('/tmp/compose.py').read().split('rs=[]')[0]); g=op('panel_b','v',np.cos)(z); U,S,V=np.linalg.svd(Y[:,1:],full_matrices=False); pc=Y[:,1:]@V[:2].T; pz=p(z);qz=q(z); T=np.c_[Y[:,0],pc]; rng=np.random.default_rng(51); splits=[(np.arange(52),np.arange(52,72))]+[(r[:52],r[52:]) for r in [rng.permutation(len(z)) for _ in range(8)]]
def ss(M):
 e=[]
 for i,j in splits:
  r=abs(M[j]@fit(M[i],Y[i])-Y[j]);e.append([r[:,0].mean(),r[:,1:].mean()]);
 return np.round([e[0],np.max(e,axis=0),np.mean(e,axis=0)],5)
D={'g':g,'z':z,'p':pz,'q':qz,'pg':p(g),'qg':q(g),'bg':op('panel_b','v',np.cos)(g)}; res=[]
for n,a in D.items():
 for k,M in [('aff',aff(a)),('quad',poly(a)),('cube',poly(a,3))]:res.append((n,k,ss(M)))
 for m,b in D.items():
  if n==m:continue
  for j,M in [('aff',aff(np.c_[a,b])),('quad_a',np.c_[poly(a),b]),('quad_both',np.c_[poly(a),poly(b)[:,1:]]),('diag',aff(np.c_[a,b,a*b])),('allcross',aff(np.c_[a,b,(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)]))]:res.append((n,m+' '+j,ss(M)))
for k in range(2):
 for n,m,r in sorted(res,key=lambda t:t[2][1,k])[:12]:print(n,m,r.tolist())
for nm,M in [('g',aff(g)),('gg',poly(g)),('gz',np.c_[poly(g),z]),('gpq',np.c_[poly(g),pz,qz]),('gpgqg',np.c_[poly(g),p(g),q(g)])]:
 c=fit(M,T); print(nm,'fit MAE',np.mean(abs(M@c-T),axis=0).round(5),'coeffs',c.round(3).T)
print('singular',S,'modes',V[:2].round(4));print('panel B coefficients',fit(np.c_[aff(u),np.cos(2.6*u)],v).round(5))
# Sparse forward search over symmetric and cyclic product monomials.
A=np.c_[g,z,pz,qz,p(g),q(g)];names=[f'{n}{j}' for n in ('g','z','p','q','pg','qg') for j in range(4)]; cols=[np.ones(len(z))]+list(A.T); nms=['1']+names
for i,j in itertools.combinations_with_replacement(range(A.shape[1]),2):cols.append(A[:,i]*A[:,j]);nms.append(names[i]+'*'+names[j])
M=np.column_stack(cols)
for t in range(3):
 ix=[0];r=T[:,t];
 for k in range(16):
  R=M-M[:,ix]@fit(M[:,ix],M); val=abs(r@R)/np.maximum(np.linalg.norm(R,axis=0),1e-8); val[ix]=-1; ix.append(np.argmax(val)); r=T[:,t]-M[:,ix]@fit(M[:,ix],T[:,t]);
  if k in (3,6,9,15):print('omp',t,k,np.mean(abs(r)).round(5),[nms[i] for i in ix])
