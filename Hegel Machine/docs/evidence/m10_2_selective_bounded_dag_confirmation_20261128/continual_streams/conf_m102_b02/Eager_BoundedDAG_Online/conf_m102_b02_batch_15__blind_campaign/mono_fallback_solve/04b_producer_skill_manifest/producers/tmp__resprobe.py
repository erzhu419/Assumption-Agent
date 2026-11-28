import numpy as np,itertools
from scipy.optimize import least_squares
p=np.load('/tmp/design.npz'); Y=np.c_[p['y'],p['m']]; N=52; z,b,c,h=[p[k][:72] for k in 'zbch']; O=np.ones(72)
def D(*a):return np.column_stack((O,*a))
def fit(F,y):return np.linalg.lstsq(F[:N],y[:N],rcond=1e-9)[0]
def proj(F,y):return F@fit(F,y)
def quad(x):return np.stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)],axis=1)
bq=quad(b); zq=quad(z); cq=quad(c); hq=quad(h)
W={'z':z,'b':b,'c':c,'h':h,'bq':bq,'zq':zq,'cq':cq,'hq':hq,'t':np.tanh(z),'g':Y[:,1:],'gq':quad(Y[:,1:])}
for j in [1,2,0]:
 y=Y[:,j]; tests=[]
 for k in range(1,4):
  for ss in itertools.combinations([s for s in W if j==0 or s not in ['g','gq']],k):
   F=D(*(W[s] for s in ss));
   if F.shape[1]>28:continue
   v=fit(F,y); e=F@v-y; tests.append((np.mean(abs(e[N:])),np.mean(abs(e[:N])),ss))
 print('linear',j,sorted(tests)[:10]);
 if j==1:
  U=np.c_[z[:,2]+z[:,3],z[:,0]-z[:,1]]
  for ss in [(),('z',),('b',),('c',),('h',),('zq',),('cq',),('z','b'),('z','c'),('z','h'),('z','c','h'),('zq','b'),('zq','h')]:
   def F(q):return D(np.tanh(U*q[:2]+q[2:]),*[W[s] for s in ss])
   q=least_squares(lambda q:(proj(F(q),y)-y)[:N], [1,1,0,0],bounds=([.01,.01,-3,-3],[10,10,3,3])).x
   e=proj(F(q),y)-y;print('paired',ss,np.round(q,3),np.round([np.mean(abs(e[:N])),np.mean(abs(e[N:]))],4),np.round(fit(F(q),y),2))
 if j==2:
  bf=[D(b,bq),D(bq),D(z,b,bq),D(b,bq,c),D(b,bq,h),D(np.c_[b[:,0]*b[:,1],b[:,2]*b[:,3],b[:,0]*b[:,3]])]; out=[]
  for nf,F in enumerate(bf):
   for k,A in [('z',z),('b',b),('c',c),('h',h)]:
    U=[(str(i),A[:,i]) for i in range(4)] + [(f'{i}{s:+d}{l}',A[:,i]+s*A[:,l]) for i,l in itertools.combinations(range(4),2) for s in [-1,1]]
    for nm,u in U:
     for mode in range(3):
      def G(q):
       den=1+q[0]*(u-q[1])**2
       if mode==0:return F/den[:,None]
       if mode==1:return D(F[:,1:]/den[:,None])
       return D(F[:,1:],1/den)
      q=least_squares(lambda q:(proj(G(q),y)-y)[:N],[.6,0],bounds=([0,-2],[20,2]),max_nfev=50).x
      e=(proj(G(q),y)-y);out.append((np.mean(abs(e[N:])),np.mean(abs(e[:N])),nf,k,nm,mode,np.round(q,3)))
  print('rational',sorted(out,key=lambda a:a[0])[:15]);print('rational fit',sorted(out,key=lambda a:a[1])[:8]);
