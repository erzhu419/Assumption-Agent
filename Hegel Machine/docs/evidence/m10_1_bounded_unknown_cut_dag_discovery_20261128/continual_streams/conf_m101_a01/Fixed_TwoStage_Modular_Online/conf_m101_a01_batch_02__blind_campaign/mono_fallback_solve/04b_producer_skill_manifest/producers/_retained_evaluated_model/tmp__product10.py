exec(open('/tmp/stages.py').read().split('train=[',1)[0])
I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72]];J=[np.setdiff1d(np.arange(72),i) for i in I]
Q=poly(x,2);wq=reg(Q,y);H=np.zeros((4,4));H[np.triu_indices(4)]=wq[5:];H=(H+H.T)/2;ev,V=np.linalg.eigh(H);print('Q directions',V[:,[0,-1]].T,'B directions',B[1:].T,flush=True)
def make(u,k,f):
 if k=='sin':return np.sin(f*u)-f*u
 if k=='cos':return np.cos(f*u)-1
 if k=='tanh':return np.tanh(f*u)-f*u
 if k=='atan':return np.arctan(f*u)-f*u
 return u**3

def fit(i,j,F0,F1,pen,link):
 # Product of affine projections; non-affine directions may enter either factor.
 A=af(x); q=poly(x[i],2); cf=reg(q,y[i]);H=np.zeros((4,4));H[np.triu_indices(4)]=cf[5:];H=(H+H.T)/2;va,ve=np.linalg.eigh(H);a=ve[:,-1]*np.sqrt(abs(va[-1]));b=ve[:,0]*np.sqrt(abs(va[0]));
 F=np.c_[A,F0]; G=np.c_[x,F1]; mi=G.shape[1]; init=np.array([np.r_[a+b,np.zeros(mi-4)],np.r_[a-b,np.zeros(mi-4)]])
 tar=y[i] if link==0 else np.arctanh(np.clip(y[i]/link,-.9999,.9999))*link
 def calc(p,idx):
  al,be=p.reshape(2,mi); M=G[idx]*1.;return np.c_[F[idx],(M@al)*(M@be)]
 def err(p):
  M=calc(p,i);v=reg(M,tar);return np.r_[M@v-tar,np.sqrt(pen)*p.reshape(2,mi)[:,4:].ravel(),.03*(np.linalg.norm(p.reshape(2,mi),axis=1)-1)]
 o=least_squares(err,init.ravel(),max_nfev=90); v=reg(calc(o.x,i),tar);p=calc(o.x,j)@v
 return np.mean(abs((p if link==0 else link*np.tanh(p/link))-y[j])), np.linalg.norm(o.fun)/len(i)**.5
Z=np.zeros((len(x),0)); cases=[('plain',Z,Z,0)]
for n in root:
 d=root[n]; p1=z[n+'1'];p2=z[n+'2'];f=np.c_[p1,p2];
 for m,r in [('c1',p1),('c2',p2),('dual',f),('trig',np.c_[make(d,'sin',1.5),make(d,'cos',1.3)]),('atan',make(d,'atan',2.1))]:
  r=r-af(x)@reg(af(x),r);r=r/np.maximum(.03,r.std(0))*.25
  cases.append((n+m,r,Z,0));cases.append((n+m+'prod',r,r,.18));
sc=[]
for n,F,G,pen in cases:
 for link in (0,1.65,2.5):
  E,R=np.array([fit(i,j,F,G,pen,link) for i,j in zip(I,J)]).T;sc.append((max(E),np.mean(E),n,link,E,R))
for a in (lambda v:v[0],lambda v:v[4][0]):
 print('TOP');print(*sorted(sc,key=a)[:13],sep='\n')
