exec(open('/tmp/profile01.py').read().split('ids=np.arange')[0])
ids=np.arange(len(y)); splits=[(ids[:52],ids[52:])]+[(ids[~np.isin(ids,v)],v) for v in np.array_split(np.random.default_rng(24).permutation(len(y)),6)]
def D(x,p,m):
 h=p[0]+x@p[1:]; A=affine(x); B=base(x)
 if m=='rational-linear': return np.column_stack([A,B/(1+h*h)[:,None]])
 if m=='rational-interaction': return np.column_stack([A,A*h[:,None]/(1+h*h)[:,None],h*h/(1+h*h)])
 if m=='rational-gate': return np.column_stack([B,A/(1+h*h)[:,None]])
 if m=='polynomial-limit':
  h=x@p[1:]; h=h/max(np.linalg.norm(p[1:]),1e-12)
  return np.column_stack([B,x*h[:,None]**2,h**4])
 if m=='atan-interaction': return np.column_stack([A,A*np.arctan(h)[:,None]])
 if m=='tanh-interaction': return np.column_stack([A,A*np.tanh(h)[:,None]])
 if m=='sin-interaction': return np.column_stack([A,A*np.sin(h)[:,None]])
 if m=='log-gate': return np.column_stack([B,A*np.log1p(h*h)[:,None]])
for m in ['rational-linear','rational-interaction','rational-gate','polynomial-limit','atan-interaction','tanh-interaction','sin-interaction','log-gate']:
 es=[]; rm=[]
 for t,v in splits:
  x=q[t]; yy=y[t]; cf=fit(base(x),yy); M=np.zeros((4,4))
  for k,(i,j) in enumerate(pairs): M[i,j]=M[j,i]=cf[k+5]/(1+(i!=j))
  e,V=np.linalg.eigh(M)
  starts=[np.r_[b,d*f] for b in [0.,.35] for d in [.45,1.3] for f in [V[:,0],V[:,-1]]]
  def resid(p):
   X=D(x,p,m);return X@fit(X,yy)-yy
  r=min((least_squares(resid,p,bounds=(-6,6),max_nfev=100,ftol=1e-7) for p in starts),key=lambda r:r.cost)
  X=D(q,r.x,m); es.append(np.mean(abs(X[v]@fit(X[t],yy)-y[v]))); rm.append(np.std(r.fun))
 print(m,'folds',np.round(es,5),'worst',round(max(es),6),'fit rmse',np.round(rm,5),flush=True)
