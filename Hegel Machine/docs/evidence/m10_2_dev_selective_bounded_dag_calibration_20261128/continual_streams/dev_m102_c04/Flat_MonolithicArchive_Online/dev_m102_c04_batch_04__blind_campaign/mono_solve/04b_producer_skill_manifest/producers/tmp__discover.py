exec(open('/tmp/probe.py').read().split('for d in range')[0])
U=s['panel_c']['u']; V=s['panel_c']['v1']; W=s['panel_c']['v2']; Bu=s['panel_b']['u']; Bv=s['panel_b']['v']; aff=lambda z:np.c_[np.ones(len(z)),z]; bw=ls(aff(Bu),Bv,rcond=None)[0]; print('B',bw)
for n,z in [('u',U),('B(u)',aff(U)@bw),('A(u)',U@M+a),('Ai(u)',(U-a)@np.linalg.inv(M))]:
 print('C1 frame',n)
 for d in [1,2,3]:
  F=aff(poly(z,d)); P=F@ls(F,V,rcond=None)[0]; print(d,np.std(P-V),'LOO',np.mean(abs((P-V)/np.maximum(1e-9,1-np.sum(np.linalg.svd(F,full_matrices=False)[0]**2,axis=1))[:,None])))
 for k in [.22,.8,1.25]:
  r=np.roll(z,-1,axis=1); F=aff(np.c_[poly(z,2),z**3,z/(1+k*abs(z)), z**2/(1+k*z**2)]); print(k,np.std(F@ls(F,V,rcond=None)[0]-V))
F=aff(poly(U,2)); print('C1 quadratic',ls(F,V,rcond=None)[0]); print('C2 fit'); k=.16; F=aff(np.c_[U,U**2/(1+k*U**2)]); print(ls(F,W,rcond=None)[0]);
H=-np.expm1(-t[:,None]*[1.795,.42]); Z=np.c_[y,ls(H,T.T,rcond=None)[0].T];
terms=[j for k in range(4) for j in combinations_with_replacement(range(4),k)]
for n,z in [('raw',x),('A inverse',q),('B raw',aff(x)@bw),('B A inverse',aff(q)@bw),('B inverse', (x-bw[0])@np.linalg.inv(bw[1:])),('B inverse A inverse',(q-bw[0])@np.linalg.inv(bw[1:]))]:
 F=aff(poly(z,3)); w=ls(F[:52],Z[:52],rcond=None)[0]; e=F@w-Z; print('frame',n,'mae',np.mean(abs(e[52:,0])),np.mean(abs(e[52:,1:]@H.T))); print('strong terms',[(terms[j],np.round(w[j],3).tolist()) for j in np.argsort(np.linalg.norm(w,axis=1))[-12:][::-1]])
