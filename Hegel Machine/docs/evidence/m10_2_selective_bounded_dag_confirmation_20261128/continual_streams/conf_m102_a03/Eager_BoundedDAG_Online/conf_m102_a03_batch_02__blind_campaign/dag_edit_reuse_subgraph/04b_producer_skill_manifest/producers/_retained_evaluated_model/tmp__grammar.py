exec(open('/tmp/probe.py').read().split('for name,x in blocks.items():')[0])
for name in ['Z','U']:
 x=blocks[name]; P=list(it.combinations_with_replacement(range(4),2))+list(it.combinations_with_replacement(range(4),3)); D=np.c_[a(x),*[np.prod(x[:,p],axis=1) for p in P]]; c=fit(D,y); print(name,'poly',[(p,round(float(v),4)) for p,v in sorted(zip(P,c[5:]),key=lambda t:-abs(t[1]))[:18]])
 for K in (5,9,15,25):
  def pred(i,j):
   A=D[:,:5]; pool=list(range(5,D.shape[1])); cols=list(range(5)); r=y[i]-A[i]@fit(A[i],y[i]);
   for _ in range(K-5):
    Q=np.linalg.qr(D[i][:,cols])[0]; R=D[i][:,pool]; R-=Q@(Q.T@R); score=np.abs(R.T@r)/np.maximum(1e-12,np.linalg.norm(R,axis=0)); cols.append(pool.pop(np.argmax(score))); A=D[:,cols]; r=y[i]-A[i]@fit(A[i],y[i]);
   return A[j]@fit(A[i],y[i])
  e=[np.mean(abs(pred(i,j)-y[j])) for i,j in splits]; print('OMP',K,np.round([e[0],np.mean(e[1:]),max(e[1:])],5))
for v in ['v1','v2']:
 p=s['panel_c']; x=p['u']; w=p[v]; print('panel',v)
 for d in [1,2]:
  D=a(x) if d==1 else np.c_[a(x),*[np.prod(x[:,p],axis=1) for p in it.combinations_with_replacement(range(4),2)]]; rr=w-D@fit(D,w); print(d,'rms',np.std(rr,axis=0),'singular',np.linalg.svd(rr,compute_uv=False))
x=z; L=a(x); B=x[:,0]*x[:,2]+x[:,1]*x[:,3]; S=x[:,0]*x[:,2]-x[:,1]*x[:,3]; A=x[:,0]*x[:,2]; C=x[:,1]*x[:,3]; base=np.c_[L,B]; mon={str(p):np.prod(x[:,p],axis=1) for d in (2,3) for p in it.combinations_with_replacement(range(4),d)}; mon.update({'sum':B,'diff':S}); print('simple',sorted([(met(np.c_[base,v])[3],k,met(np.c_[base,v])) for k,v in mon.items()])[:8]); print('sum diff',met(np.c_[L,B,S]))
for name in ('Z','U','v1','v2'):
 x=blocks[name]; qs={str(j):x[:,j] for j in range(4)}; qs.update({f'{j}{sgn}{k}':x[:,j]+(1 if sgn=='+' else -1)*x[:,k] for j,k in it.combinations(range(4),2) for sgn in '+-'}); qs.update({'norm':np.linalg.norm(x,axis=1),'sumprod':x[:,0]*x[:,2]+x[:,1]*x[:,3],'diffprod':x[:,0]*x[:,2]-x[:,1]*x[:,3]}); results=[]
 for k,q in qs.items():
  for lam in (.22,.5,1.,1.25,1.5625,2.,4.):
   g=1/(1+lam*q*q)
   for typ,D in [('B',np.c_[L,B*g]),('AB',np.c_[L,A*g,C*g]),('LIN',np.c_[L,B,z*g[:,None]]),('ALL',np.c_[L,z*g[:,None],B*g]),('MIX',np.c_[L,B,A*g,C*g]),('WHOLE',base*g[:,None]),('CUB',np.c_[base,z*q[:,None]**2])]:
    r=y-D@fit(D,y); results.append((np.mean(abs(r)),k,lam,typ,D))
 print('rational',name,[(round(m,6),k,p,t,met(D)[:3]) for m,k,p,t,D in sorted(results,key=lambda e:e[0])[:10]])
