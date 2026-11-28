import numpy as np,itertools as I
from scipy.optimize import least_squares
p=np.load('/tmp/design.npz'); z,b,c,h=[p[k][:72] for k in 'zbch']; M=p['m']; N=52; O=np.ones(72); T=p['T']; f=p['f']; y=p['y'];
D=lambda *a:np.column_stack((O,*a)); fit=lambda F,y:np.linalg.lstsq(F[:N],y[:N],rcond=1e-8)[0];
def Q(x):return np.stack([x[:,i]*x[:,j] for i,j in I.combinations_with_replacement(range(x.shape[1]),2)],1)
bq=Q(b); sel=[1,3,5,8]; base=[D(z),D(bq[:,sel])]; u0=base[0]@fit(base[0],M[:,0]); u1=base[1]@fit(base[1],M[:,1]); print('basecoef',*[fit(g,M[:,j]) for j,g in enumerate(base)],flush=True)
P={};
for k,A in zip('zbch',[z,b,c,h]):
 for i in range(4): P[k+str(i)]=A[:,i]
 for i,j in I.combinations(range(4),2):
  for q in [-1,1,-.5,.5,-2,2]:P[f'{k}{i}{q:+g}{j}']=A[:,i]+q*A[:,j]
  P[f'{k}{i}*{j}']=A[:,i]*A[:,j]
 for q in [-1,1]:P[f'{k}group{q}']=A[:,0]+q*A[:,1]+A[:,2]+q*A[:,3]
 P[k+'norm']=np.sum(A*A,1)
for k,A in [('bq',bq),('zq',Q(z)),('z3',z**3)]:
 for i in range(A.shape[1]):P[k+str(i)]=A[:,i]
P.update(u0=u0,u1=u1)
for k,A,B in [('zb',z,b),('zh',z,h),('bh',b,h),('zc',z,c)]:
 for i,j in I.product(range(4),repeat=2):P[f'{k}{i}*{j}']=A[:,i]*B[:,j]
for j,F in enumerate(base):
 yy=M[:,j]; rows=[]
 for nm,u in P.items():
  for q in [.22,.6,1.25,2.15,3.2,4.5]:
   U=q*u
   for label,v in dict(s=np.sin(U),c=np.cos(U),t=np.tanh(U),r=U/(1+U*U),g=1/(1+U*U),p=np.maximum(U,0),a=np.abs(U),x=u).items():
    for kind,G in [('add',np.c_[F,v]),('all',D(F*v[:,None])),('more',np.c_[F,F*v[:,None]])]:
     e=G@fit(G,yy)-yy;rows.append((np.mean(abs(e[:N])),np.mean(abs(e[N:])),nm,q,label,kind))
 print('GRAM',j,'fit',sorted(rows)[:12], 'cv',sorted(rows,key=lambda a:a[1])[:8],flush=True)
 for X0,nm in [(z,'z'),(b,'b'),(bq[:,sel],'bqs'),(bq,'bq'),(h,'h'),(np.c_[z,b],'zb'),(np.c_[z,h],'zh')]:
  if j==0 and nm.startswith('bq'):continue
  X=D(X0); results=[]
  for typ in ('g','s','c','t','r','soft','q','exp','hinge'):
   for kind in ['add','mix','prod']:
    def gf(w):
     a=np.clip(X@w,-35,35); v={'g':lambda:1/(1+a*a),'s':lambda:np.sin(a),'c':lambda:np.cos(a),'t':lambda:np.tanh(a),'r':lambda:a/(1+a*a),'soft':lambda:a/np.sqrt(1+a*a),'q':lambda:a*a,'exp':lambda:np.exp(-a*a),'hinge':lambda:np.maximum(a,0)}[typ]();return {'add':lambda:np.c_[F,v],'mix':lambda:np.c_[F,F*v[:,None]],'prod':lambda:D(F*v[:,None])}[kind]()
    def err(w):
     G=gf(w);return (G@fit(G,yy)-yy)[:N]
    rng=np.random.default_rng(445); best=None
    for w in np.r_[np.eye(X.shape[1])*1.25,rng.normal(0,.6,(7,X.shape[1]))]:
     r=least_squares(err,w,bounds=(-8,8),max_nfev=75); score=np.mean(r.fun**2)
     if best is None or score<best[0]:best=(score,r.x)
    G=gf(best[1]);e=G@fit(G,yy)-yy;results.append((np.abs(e[:N]).mean(),np.abs(e[N:]).mean(),typ,kind,np.round(best[1],3).tolist()))
  print('PROJ',j,nm,sorted(results)[:4],'cv',sorted(results,key=lambda a:a[1])[:2],flush=True)
