exec(open('/tmp/explore.py').read().split('a=s[')[0])
from scipy.linalg import qr
p=s['panel_a']; keep=np.linalg.norm(p['v']-p['u'],axis=1)<2; a=np.array([ls(poly(p['u'][keep,j:j+1],1),p['v'][keep,j]) for j in range(4)]); p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); p=s['panel_c']; u=p['u']; V=p['v1']; W=p['v2'];
print('C affine matrix and output ranges',ls(poly(u,1),V),V.min(0),V.max(0)); print('C v2 ranges',W.min(0),W.max(0));
def prof(X,Y,D):
 for offset in [0,1,2,3]:
  U=X[:,(np.arange(4)+offset)%4];
  for name,f in [('sin',np.sin),('tanh',np.tanh),('identity',lambda v:v)]:
   if name=='identity' and offset>0: continue
   def calc(r):
    Q=np.c_[D,f(r[0]*U+r[1])]; return Q@ls(Q,Y)-Y
   rr=least_squares(lambda r:calc(r).ravel(),[1.3,.05],bounds=([.05,-2],[6,2]),max_nfev=50); print(name,offset,np.std(rr.fun),rr.x)
  if offset==0: break
D=np.c_[poly(u,1),u**2,u*np.roll(u,1,axis=1),u[:,0]*u[:,2],u[:,1]*u[:,3]]
for nm,X in [('u',u),('v1',V)]:
 print('C mapping',nm,'into v2'); prof(X,W,D)
print('C mapping u into v1'); prof(u,V,poly(u,1))
# Use independent held-out rows to compare alternative structure and stabilize the state fit.
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
rng=np.random.default_rng(512); splits=[np.arange(56,72)]+[k for _ in range(3) for k in np.array_split(rng.permutation(n),5)]
mode=np.linalg.svd(z,full_matrices=False)[2][0]; out=np.c_[y,z@mode];
def score(K):
 e=[]
 for ix in splits:
  it=np.setdiff1d(np.arange(n),ix); e.append(np.mean(np.abs(K[np.ix_(ix,it)]@cho_solve(cho_factor(K[np.ix_(it,it)]),out[it])-out[ix]),axis=0))
 return np.asarray(e)
for ca,X0 in [('raw',x),('A+',x*a[:,1]+a[:,0]),('A-',(x-a[:,0])/a[:,1])]:
 for cb,X in [('raw',X0),('B+',poly(X0,1)@B),('B-',(X0-B[0])@np.linalg.inv(B[1:]))]:
  P=poly(X,2); K=P@P.T; d=cdist(X,X,'sqeuclidean'); configs=[]
  for kind in ['se','mat','dot']:
   for l in [1.25,1.75,2.5,3.5,5]:
    r=np.sqrt(5*d)/l; G=(1+r+r*r/3)*np.exp(-r) if kind=='mat' else np.exp(-d/(2*l*l));
    if kind=='dot': G*=1+X@X.T
    for k in [.2,2,10]:
     for noise in [.00015,.001,.006]:
      E=score(K+k*G+noise*np.eye(n)); configs.append((E.mean(0)+.4*E.max(0),E[0],E.mean(0),E.max(0),(kind,l,k,noise)))
  print(ca,cb,'best y and state',*[min(configs,key=lambda r:r[0][j])[1:] for j in range(2)])
