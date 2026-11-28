exec(open('/tmp/grammar.py').read().split('_,_,V=')[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
V=np.linalg.svd(t[:n],full_matrices=False)[2]; a=t@V[0]
X=A(x); Id=np.eye(n); rng=np.random.default_rng(801); splits=[(np.arange(n),np.arange(n,len(x)))]+[(p[:n],p[n:]) for p in [rng.permutation(len(x)) for _ in range(5)]]
def blocks(W,mode):
 K=[]; tags=[]
 def put(tag,G): K.append(G); tags.append(tag)
 def feat(tag,f): f=np.reshape(f,(len(x),-1)); put(tag,f@f.T)
 if mode in ('core','mix','poly'):
  for key,w in W.items():
   if mode=='core' and key not in ('x','z','h','bz'):continue
   if mode=='poly' and key not in ('x','z','h','bz'):continue
   for d in range(1,7):
    feat((key,'p',d),np.stack([np.prod(w[:,ii],1) for ii in combinations_with_replacement(range(4),d)],1))
   if mode=='poly':continue
   D=(w[:,None,:]-w[None,:,:])**2
   for le in [.48,.85,1.5,2.6]:
    for o in [1,2,3,4]:
     if mode=='core' and o not in (1,2,4):continue
     G=sum(np.exp(-np.sum(D[:,:,ii],axis=2)/le**2/2) for ii in __import__('itertools').combinations(range(4),o)); put((key,'rbf',le,o),G)
   for j in [.22,1.25]:
    i,k=np.triu_indices(4,1); d=np.c_[w,w[:,i]+w[:,k],w[:,i]-w[:,k]]
    for tag,f in [('psoft',soft(d,j)),('prad',rad(d,j)),('prat',d/(1+j*d*d)),('psi',np.sin(j*d)),('pth',np.tanh(j*d))]:
     feat((key,tag,j),f); feat((key,'x'+tag,j),f[:,:,None]*w[:,None,:]);
 if mode=='ard':
  for key,w in W.items():
   if key not in ('x','z','h','bz','zc0','zc1','bzc0','bzc1'): continue
   for d in range(1,6):
    for ii in combinations_with_replacement(range(4),d):
     if d>2 and len(set(ii))>2: continue
     feat((key,'p',ii),np.prod(w[:,ii],1))
   for j in [.22,1.25]:
    i,k=np.triu_indices(4,1); d=np.c_[w,w[:,i]+w[:,k],w[:,i]-w[:,k]]
    for tag,F in [('prad',rad(d,j)),('prat',d/(1+j*d*d)),('psi',np.sin(j*d)),('pth',np.tanh(j*d))]:
     for p,f in enumerate(F.T): feat((key,tag,j,p),f)
 return tags,np.array(K)
for mode in ['poly','core','mix','ard']:
 tags,G=blocks(W,mode); print('mode',mode,len(G),flush=True)
 for reg in [0,1,5]:
  for r in [y,a]:
   ev=[]
   for tr,va in splits:
    Q=np.linalg.qr(X[tr])[0]; P=np.eye(n)-Q@Q.T; H=P@G[:,tr[:,None],tr]@P; H/=np.maximum(np.sqrt(np.mean(H**2,axis=(1,2))),1e-14)[:,None,None]; norm=np.maximum(np.sqrt(np.mean((P@G[:,tr[:,None],tr]@P)**2,axis=(1,2))),1e-14); keep=norm>1e-10; H=H[keep]; norm=norm[keep]; rr=P@r[tr]; B=G[keep][:,va[:,None],tr]; B=(B-X[va]@np.linalg.pinv(X[tr])@G[keep][:,tr[:,None],tr])@P/norm[:,None,None]; B0=X[va]@ls(X[tr],r[tr]); H=np.concatenate((H,Id[None])); Hf=H.reshape(len(H),-1)
    def loss(th):
     u=np.exp(th); f=cho_factor((u@Hf).reshape(n,n)); al=cho_solve(f,rr); inv=cho_solve(f,Id); val=.5*rr@al+np.log(np.diag(f[0])).sum()+reg*u[:-1].sum(); grad=.5*u*(Hf@(inv-np.outer(al,al)).ravel()); grad[:-1]+=reg*u[:-1]; return val,grad
    init=np.r_[np.full(len(H)-1,-6-np.log(len(H)-1)), -9]; opt=minimize(loss,init,jac=True,method='L-BFGS-B',bounds=[(-24,1)]*(len(H)-1)+[(-15,-3)],options={'maxiter':220,'ftol':1e-8}); u=np.exp(opt.x); al=cho_solve(cho_factor((u@Hf).reshape(n,n)),rr); pred=B0+(u[:-1]@B.reshape(len(B),-1)).reshape(len(va),n)@al; ev.append(np.mean(abs(r[va]-pred)))
    if len(ev)==1 and reg==1:print('top',[(np.array(tags,dtype=object)[i],round(u[j],5)) for j,i in [(j,np.flatnonzero(keep)[j]) for j in np.argsort(u[:-1])[-5:][::-1]]],flush=True)
   print(reg,'y' if r is y else 'a',np.round(ev,5),'avg',np.mean(ev),flush=True)
