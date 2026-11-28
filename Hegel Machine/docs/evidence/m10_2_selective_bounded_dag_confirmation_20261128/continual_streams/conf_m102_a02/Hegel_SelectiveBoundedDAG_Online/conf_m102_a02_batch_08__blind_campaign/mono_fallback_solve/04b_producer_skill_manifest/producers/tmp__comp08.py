exec(open('/tmp/read08.py').read().split("print('C range'")[0]); from scipy.optimize import minimize_scalar
h=q[:,2]; g=h*h; N=np.ones(len(q));
def fitcv(build,bd=(0,1)):
 e=[]; ps=[]; rms=[]
 for i,j in folds:
  def loss(p):
   f=build(p)[i];return np.mean((f@ls(f,y[i])-y[i])**2)
  p=minimize_scalar(loss,bounds=bd,method='bounded').x; f=build(p);e.append(err(f[j]@ls(f[i],y[i])-y[j]));ps.append(p);rms.append(loss(p)**.5)
 return np.mean(e[1:]),fmt(e),fmt(ps),fmt(rms)
opts=[]
for name,H in [('q',q),('b',b),('c',c),('qb',np.c_[q,b])]:
 for j in range(H.shape[1]):
  for k in range(j,H.shape[1]):
   for sg in ([1] if k==j else [1,-1]):
    p0=(H[:,j] if j==k else H[:,j]+sg*H[:,k])**2
    for tag in ('B','B&C','diff','diff+C'):
     def build(p):
      f=1/(1+p*p0); f=f[:,None];
      if tag=='B': return np.c_[N,C,b*f]
      if tag=='B&C':return np.c_[N,np.c_[b,C]*f]
      if tag=='diff':return np.c_[base,b*((1-f)/max(p,1e-9))]
      return np.c_[base,np.c_[b,C]*((1-f)/max(p,1e-9))]
     opts.append((f'{name} {j} {sg:+} {k} {tag}',fitcv(build)))
for k,v in sorted(opts,key=lambda a:a[1][0])[:22]:print(k,v)
for tag,branch in [('base',base),('q2^2*b',np.c_[base,b*g[:,None]]),('q2^2*b*C',np.c_[base,b*g[:,None],C*g]),('q2^2*(b,q)',np.c_[base,np.c_[b,q]*g[:,None]]),('q2^2*b, q^3',np.c_[base,b*g[:,None],q**3]),('gated C + panel-B',np.c_[base,C*g]),('linear and poly gate',np.c_[base,b*h[:,None],b*g[:,None]])]:print(tag,fmt(cv(branch)),fmt(ls(branch[:52],y[:52])),err(branch[:52]@ls(branch[:52],y[:52])-y[:52]))
print('gate and q geometry',np.quantile(np.abs(q),[0,.5,.9,1],axis=0), 'raw y noise high freq',np.median(np.abs(np.diff(z,n=3,axis=1)),axis=0))
