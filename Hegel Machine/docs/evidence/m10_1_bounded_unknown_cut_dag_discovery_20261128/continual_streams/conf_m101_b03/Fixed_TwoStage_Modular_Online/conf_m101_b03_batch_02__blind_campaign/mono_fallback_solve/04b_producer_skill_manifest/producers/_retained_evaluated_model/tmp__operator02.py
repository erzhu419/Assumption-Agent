exec(open('/tmp/probe02.py').read().split('for name,Z')[0])
C=s['panel_c']; u=C['u']; cyc=tuple((i,(i+1)%4) for i in range(4))
def lift(v): return np.c_[np.ones(len(v)),v,*[v[:,i]*v[:,j] for i,j in cyc]]
M=np.linalg.lstsq(lift(u),C['v1'],rcond=None)[0]; print('C1 quadratic residual',np.sqrt(np.mean((lift(u)@M-C['v1'])**2)), 'coeff',np.round(M,3))
fun={'cos':lambda x,k:np.cos(k*x),'rat':lambda x,k:1/(1+k*x*x),'root':lambda x,k:np.sqrt(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'quadratic':lambda x,k:x*x}
for fname,fn in fun.items():
 def feat(v,k):return np.c_[np.ones(len(v)),v,fn(v,k)]
 def fit(k):return np.linalg.lstsq(feat(u,k),C['v2'],rcond=None)[0]
 def err(k):return (feat(u,k)@fit(k)-C['v2']).ravel()
 rr=[least_squares(err,np.full(4,k),bounds=(.08,6.),max_nfev=180) for k in (.4,1.3,2.5,4.)]; r=min(rr,key=lambda r:np.linalg.norm(r.fun)); print('C2',fname,np.round(r.x,4),np.sqrt(np.mean(r.fun**2)), np.round(fit(r.x)[5:],2));
 if fname=='cos': K=r.x; D=fit(K)
for key,zz in [('z',z),('Bz',z@B[1:]+B[0]),('Bi',(z-B[0])@np.linalg.inv(B[1:]))]:
 c1=lift(zz)@M; c2=np.c_[np.ones(len(zz)),zz,np.cos(zz*K)]@D
 for nm,f in [('z+C1',np.c_[np.ones(len(z)),z,c1]),('z+C2',np.c_[np.ones(len(z)),z,c2]),('z+C1+C2',np.c_[np.ones(len(z)),z,c1,c2]),('cyc+cos',np.c_[lift(z),np.cos(zz*K)]),('cyc+sq+cos',np.c_[lift(z),z*z,np.cos(zz*K)]),('quad+cos',np.c_[np.ones(len(z)),poly(z,2)[0],np.cos(zz*K)]),('quad+cos+c1 nonlinear',np.c_[np.ones(len(z)),poly(z,2)[0],np.tanh(c1),np.sin(c1),1/(1+c1*c1)])]:
  pred=f@np.linalg.lstsq(f[a],Y[a],rcond=1e-7)[0]; print(key,nm,'fit',np.round(np.mean(abs(pred[a]-Y[a]),axis=0),4),'val',score(pred[q]))
for nm,f in [('cyc',lift(z)),('cyc+sq',np.c_[lift(z),z*z]),('quad',np.c_[np.ones(len(z)),poly(z,2)[0]])]:
 m=np.c_[f,Y[:,1:]]; coef=np.linalg.lstsq(m[a],y[a],rcond=None)[0]; print('oracle',nm,'coeff',np.round(coef,4),'fit, val',*[np.mean(abs(m[ix]@coef-y[ix])) for ix in (a,q)])
# Find which residual components require the same missing feature.
for nm,f in [('cyc',lift(z)),('quad',np.c_[np.ones(len(z)),poly(z,2)[0]])]:
 res=Y-f@np.linalg.lstsq(f[a],Y[a],rcond=None)[0]; print('residual spectrum',nm,np.linalg.svd(res[a],compute_uv=False),np.corrcoef(res[a].T).round(3))
