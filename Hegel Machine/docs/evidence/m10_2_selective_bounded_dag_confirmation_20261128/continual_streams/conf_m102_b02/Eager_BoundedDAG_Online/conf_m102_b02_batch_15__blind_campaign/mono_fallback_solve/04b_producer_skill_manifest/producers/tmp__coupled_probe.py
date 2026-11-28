import itertools as I,numpy as np
from scipy.optimize import least_squares
p=np.load('/tmp/design.npz'); z,b,c,h=[p[k][:72] for k in 'zbch']; y=p['y']; M=p['m']; N=52; f=p['f']; T=p['T']
def D(*a):return np.column_stack((np.ones(72),*a))
def Q(x):return np.array([x[:,i]*x[:,j] for i,j in I.combinations_with_replacement(range(x.shape[1]),2)]).T
def fit(F,y):return np.linalg.lstsq(F[:N],y[:N],rcond=1e-9)[0]
print('scalar coupling',fit(D(z,Q(b),M),y))
for label,F in [('slow z',D(z)),('fast bq',D(Q(b)))]:
 j=int(label.startswith('fast')); w=fit(F,M[:,j]); print(label,w); print('samples [z b target fitted residual]',np.round(np.c_[z,b,M[:,j],F@w,M[:,j]-F@w][np.argsort(M[:,j])][::3],3))
# Learn alternative temporal-mode rotation jointly with structured heads.
for j in range(2):
 pool=[]
 for k,A in zip('zbch',[z,b,c,h]):
  P=[(str(i),A[:,i]) for i in range(4)]+[(f'{i}{s:+d}{l}',A[:,i]+s*A[:,l]) for i,l in I.combinations(range(4),2) for s in [-1,1]]
  for n,u in P:
   for q in [.5,1,1.25,1.7,2.15,2.6,3.5]:
    for op,g in [('s',np.sin),('c',np.cos),('t',np.tanh),('r',lambda u:u/(1+u*u)),('a',np.arctan),('e',lambda u:np.exp(-u*u)),('f',lambda u:u/np.sqrt(1+u*u))]:pool.append((f'{k}{n}:{op}{q}',g(q*u)))
 for name,F in ([('z',D(z)),('zh',D(z,h)),('zq',D(z,Q(z)))] if j==0 else [('bq',D(Q(b))),('bbq',D(b,Q(b))),('bp',D(b[:,0]*b[:,1],b[:,2]*b[:,3],b[:,0]*b[:,3]))]):
  target=M[:,j]; G=np.column_stack([v for _,v in pool]); print('HEAD',j,name)
  for mix in [False,True]:
   H=np.c_[F,M[:,1-j]] if mix else F; ids=[]
   for step in range(6):
    R=target-H@fit(H,target); Z=G-H@fit(H,G); norm=np.linalg.norm(Z[:N],axis=0); scores=np.abs(Z[:N].T@R[:N])/np.maximum(norm,1e-8); scores[norm<1e-6]=0; i=scores.argmax();ids.append(pool[i][0]);H=np.c_[H,G[:,i]]
    e=H@fit(H,target)-target;print(mix,step,ids[-1],np.round([np.mean(abs(e[:N])),np.mean(abs(e[N:])),fit(H,target)[F.shape[1]] if mix else 0],4))
# Test saturation of a quadratic projection, with or without a second-mode admixture.
for j in range(2):
 for nm,X in [('z',z),('zq',np.c_[z,Q(z)]),('bq',Q(b)),('b',b),('h',h)]:
  if j==0 and nm in ('bq','b') or j==1 and nm in ('z','h'):continue
  for typ in ['t','s','a','r','f','linear','sq']:
   for mix in [False,True]:
    X1=D(X); oth=M[:,1-j]; yy=M[:,j]; fn={'t':np.tanh,'s':np.sin,'a':np.arctan,'r':lambda u:u/(1+u*u),'f':lambda u:u/np.sqrt(1+u*u),'linear':lambda u:u,'sq':lambda u:u*u}[typ]; w=fit(X1,yy)
    def fun(q):
     return q[-3]+q[-2]*fn(X1@q[:-3])+q[-1]*oth*mix
    res=least_squares(lambda q:(fun(q)-yy)[:N],np.r_[w*.5,0,2,0],max_nfev=150); e=fun(res.x)-yy
    if np.mean(abs(e[N:]))<.12:print('SAT',j,nm,typ,mix,np.round([np.mean(abs(e[:N])),np.mean(abs(e[N:]))],4),np.round(res.x,3))
