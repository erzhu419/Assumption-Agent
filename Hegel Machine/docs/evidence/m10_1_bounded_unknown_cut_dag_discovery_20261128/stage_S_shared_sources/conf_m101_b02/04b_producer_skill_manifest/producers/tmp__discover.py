import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aug(x): return np.c_[np.ones(len(x)),x]
def fit(g,y): return np.linalg.lstsq(g,y,rcond=None)[0]
def affine(d,trim=False):
 g=aug(d['u']);v=d['v'];m=np.ones(len(g),bool)
 if trim:
  e=np.linalg.norm(v-g@fit(g,v),axis=1);m=e<4*np.median(e)
 b=fit(g[m],v[m]);return lambda x:aug(x)@b
A=affine(s['panel_a'],True);B=affine(s['panel_b']);u=s['panel_c']['u'];v=s['panel_c']['v1'];w=s['panel_c']['v2'];np.set_printoptions(precision=6,suppress=True,linewidth=140)
def loo(g,y):
 e=y-g@fit(g,y);h=(np.linalg.qr(g,mode='reduced')[0]**2).sum(1);return np.mean(e**2)**.5,np.mean((e/(1-h[:,None]))**2)**.5
maps=[]
for label,f in {'rat':lambda x,a:x/(1+a*x*x),'sqrt':lambda x,a:x/np.sqrt(1+a*x*x),'tanh':lambda x,a:np.tanh(a*x),'atan':lambda x,a:np.arctan(a*x),'asinh':lambda x,a:np.arcsinh(a*x),'erf':lambda x,a:__import__('scipy').special.erf(a*x),'log':lambda x,a:np.sign(x)*np.log1p(a*np.abs(x))}.items():
 for model in ('basic','linear','shift','linearshift'):
  n=8 if 'shift' in model else 4
  def gram(x,p,f=f,model=model):
   z=x+p[4:] if len(p)==8 else x; z=f(z,np.exp(p[:4]));return aug(np.c_[x,z] if 'linear' in model else z)
  def resid(p):
   g=gram(u,p);return (g@fit(g,v)-v).ravel()
  q=least_squares(resid,np.r_[np.zeros(4),np.zeros(n-4)],bounds=(np.r_[np.repeat(-4,4),np.repeat(-.7,n-4)],np.r_[np.repeat(4,4),np.repeat(.7,n-4)]),max_nfev=250);g=gram(u,q.x);m=loo(g,v);maps.append((m,label,model,q.x,lambda x,g=gram,b=fit(g,v),p=q.x:g(x,p)@b))
for m,k,j,p,_ in sorted(maps,key=lambda r:r[0][1])[:12]:print('C1',k,j,m,np.exp(p[:4]),p[4:])
print('v2 monomials')
cols=list(combinations_with_replacement(range(4),2));g=np.c_[aug(u),*[u[:,i]*u[:,j] for i,j in cols]];print(loo(g,w)); print(list(zip(['const',0,1,2,3]+cols,fit(g,w))))
for tag,cols2 in [('cyclic',[(0,1),(1,2),(2,3),(0,3)]),('square',[]),('all',[(i,j) for i,j in cols if i!=j])]:
 def gv(x,cols2=cols2):return np.c_[aug(x),x*x,*[x[:,i]*x[:,j] for i,j in cols2]]
 G=gv(u); print('C2',tag,loo(G,w));
 for k in range(G.shape[1]):
  H=np.delete(G,k,axis=1);r=loo(H,w)
  if r[1]<.012:print('drop',k,r)
C=sorted(maps,key=lambda r:r[0][1])[0][-1]; x=s['train']['x'];y=s['train']['y'];z=s['train']['traj'];Y=np.c_[y,z]; rr=[]
for name,F in {'raw':x,'B':B(x),'AB':A(B(x)),'BA':B(A(x)),'abi':(B(x)-A(np.zeros_like(x)))/(A(np.ones_like(x))-A(np.zeros_like(x)))}.items():
 c=C(F); d=np.c_[aug(F),*[F[:,i]*F[:,j] for i,j in cols]]@fit(g,w)
 parts={'raw':F,'C':c,'D':d,'pair':F*c,'C2':c*c,'raw2':F*F,'cross':np.column_stack([c[:,i]*c[:,j] for i,j in cols if i!=j]),'mix':np.column_stack([F[:,i]*c[:,j] for i in range(4) for j in range(4) if i!=j]),'rawcross':np.column_stack([F[:,i]*F[:,j] for i,j in cols if i!=j])}
 keys=list(parts)
 for mask in range(1,1<<len(keys)):
  inds=[keys[i] for i in range(len(keys)) if mask>>i&1];G=aug(np.column_stack([parts[k] for k in inds]));tr=G[:52];
  if G.shape[1]>36: continue
  for l in [.0001,.003,.03]:
   a=tr.T@tr+l*np.eye(tr.shape[1]);p=G[52:]@np.linalg.solve(a,tr.T@Y[:52]);e=np.mean(abs(p-Y[52:]),axis=0);rr.append((e[0],np.mean(e[1:]),e[-1],name,','.join(inds),l))
for k in range(2):print('top',k,*sorted(rr,key=lambda r:r[k])[:8],sep='\n')
