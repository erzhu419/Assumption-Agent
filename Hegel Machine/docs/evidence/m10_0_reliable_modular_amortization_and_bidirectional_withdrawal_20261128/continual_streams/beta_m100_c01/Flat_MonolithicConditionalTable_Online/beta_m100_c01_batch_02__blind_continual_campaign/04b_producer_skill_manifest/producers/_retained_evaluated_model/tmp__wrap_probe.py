exec(open('/tmp/wall_probe.py').read().split('for proj in (False,True):')[0])
base=axis
def axis(a,n,bc):
 k=s['kappa_'+a]; h=s['L'+a]/(n-1); v=s['vel_'+a]; r=s['robin_'+a]/h
 A=base(a,n,'NN').toarray()
 for i,j,t in [(0,1,bc[0]),(n-1,n-2,bc[1])]:
  p=n-1-i; d=k[i]/h**2; f=(k[i]+k[j])/(2*h*h); fw=(k[i]+k[p])/(2*h*h); vi=(v[i] if i==0 else -v[i])/h
  if isinstance(t,str):
   near,wrap={'row':(d,d),'avg':(f,fw),'mix':(d,fw),'period':(f+max(-vi,0),fw+max(vi,0)),'rowadv':(d+max(-vi,0),d+max(vi,0)),'wraponly':(0,fw),'no':(d,0),'dup':(f,0)}[t]; leak=0
  else:
   near,wrap,leak=t*np.array([d,d,r]);
  A[i,i]=near+wrap+leak;A[i,j]=-near;A[i,p]=-wrap
 return csc_matrix(A)
def score(E):
 return np.sqrt(np.mean(E.reshape(3,-1,len(p))**2,axis=1)) if False else np.sqrt(np.mean(E.reshape(3,-1,W.shape[1])**2,axis=1))
for proj in (False,True):
 print('projection',proj,flush=True)
 types=['row','avg','mix','period','rowadv','wraponly','no','dup']; results=[]
 for bx,by in itertools.product(types,repeat=2):
  E=residual((bx,bx,by,by),proj);results.append((np.linalg.norm(E),bx,by,rms(E)))
 for v in sorted(results,key=lambda r:r[0])[:6]: print(v,flush=True)
 for held in (-1,0,1,2):
  tr=np.arange(W.shape[1])!=held
  fit=least_squares(lambda t:residual(t.reshape(4,3),proj)[:,tr].ravel(),np.tile([1.,1.,0.],4),gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=160)
  E=residual(fit.x.reshape(4,3),proj)
  print('free-stencil fit',held,fit.x.reshape(4,3).round(6),rms(E),score(E),flush=True)
