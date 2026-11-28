from batch_solver import *
import pickle,itertools
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train']; z,q=coords(s,t['x']);y=t['y']; T=t['traj']; n=len(y); I=np.arange(52); J=np.arange(52,n); ts=s['eval_times']; B=np.c_[-np.expm1(-ts),ts]; A=ls(B,T.T).T; Y=np.c_[y,A];
warp={'sin':np.sin,'cos':lambda x:np.cos(x)-1,'tanh':np.tanh,'rat':lambda x:x/(1+x*x)}
ops={}
for nm,k,f,i in [('B','v','sin',[-.4,2.9]),('C','v1','tanh',[2,2.27]),('D','v2','cos',[-.65,1.29])]:
 p=s['panel_b' if nm=='B' else 'panel_c'];u=p['u'];v=p[k]; g=warp[f]; d=lambda u,p,g=g:aff(u+p[0]*g(p[1]*u)); r=least_squares(lambda p:(d(u,p)@ls(d(u,p),v)-v).ravel(),i,max_nfev=45); p=r.x;c=ls(d(u,p),v);ops[nm]=lambda u,d=d,p=p,c=c:d(u,p)@c
 print(nm,p,np.sqrt(np.mean((d(u,p)@c-v)**2)))
F={'z':z,'x':t['x'],'q':q};L=aff(q);Q=quad(q);base=np.c_[L,Q];
for n0 in ['z','q','x']:
 for nm,op in ops.items():F[n0+nm]=op(F[n0])
for rn,w in list(F.items()):
 if rn=='q' or len(rn)==2:
  for nm,op in ops.items():F[rn+nm]=op(w)
for k in ['q','z','qC','qD','zC','zD']:
 for nm in ['q','z']:
  if nm==k:continue
  for a in [-1,-.5,-.22,.22,.5,1]:F[f'{nm}+{a}{k}']=F[nm]+a*F[k]
def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
results=[]
def test(k,d):
 d=np.asarray(d); d=d-d[I].mean(0);d=d/np.maximum(d[I].std(0),1e-10);testk(k,d@d.T)
def testk(k,K):
 Ki=K[np.ix_(I,I)]; mj=K[:,I].mean(1);mi=mj[I]; K=K[:,I]-mi[None,:]-mj[:,None]+mi.mean(); ei,U=np.linalg.eigh(K[I]); ei=np.maximum(ei,0); yp=U.T@(Y[I]-Y[I].mean(0)); E=[]
 for a in [.00001,.0003,.003,.03,.2,1,5]:
  D=1/(ei+a); aa=U@(D[:,None]*yp); ey=(aa/np.maximum(1e-12,(U*U)@D-1/(len(I)*a))[:,None]); p=K@aa+Y[I].mean(0);E.append((abs(ey).mean(0),abs(p[J]-Y[J]).mean(0),a,p))
 e=min(E,key=lambda r:r[0][0]); et=min(E,key=lambda r:r[0][1]);results.append((k,e,et))
for k,f in F.items():
 for lk,v in [('q',q),('f',f),('qz',np.c_[q,z]),('qf',np.c_[q,f])]:
  test((k,lk,'quad'),np.c_[v,quad(f)]);test((k,lk,'comb'),np.c_[v,f,quad(f),Q,prod(q,f)])
 for l in [.7,1.3,2.3,4.,7.]:
  fn=f/np.maximum(f[I].std(0),.05); R=cdist(fn,fn,'euclidean');H=1.732*R/l;D=np.c_[q,quad(q),z,quad(z)];D=(D-D[I].mean(0))/np.maximum(D[I].std(0),1e-10);K0=D@D.T/len(D.T)
  testk((k,l,'matern'),K0+(1+H)*np.exp(-H));testk((k,l,'rbf'),K0+np.exp(-.5*(R/l)**2))
for k in ['q','z','qC','qD']:
 f=F[k]; h=quad(f); pairs=[f[:,i]+a*f[:,j] for i,j in itertools.combinations(range(4),2) for a in [-1,-.5,.5,1]]; pairs=np.c_[f,np.array(pairs).T];
 for nm,g in warp.items():
  for w in [.4,1.25,2.15,2.9,4.3,6.]:
   a=g(w*pairs); test((k,nm,w,'pair'),np.c_[q,Q,a]);test((k,nm,w,'pairprod'),np.c_[q,Q,a,prod(q,a)]);
# Black-box multioutput ridge networks with a shared scalar nonlinear correction on top of a compact quadratic.
non=[]
for kn in ['q','z','qC','qD']:
 h=F[kn]; hl=aff(h); hq=np.c_[hl,quad(h)];
 for typ in ['rad','metric','quadmetric','norm','sinindex','tanhindex','rtindex','orthmix']:
  for form in([0,1,2] if typ in ['rad','metric','quadmetric','norm'] else [0,1]):
   X=np.c_[L,Q] if form!=2 else np.c_[L,quad(h)]
   def ds(p):
    if typ=='rad': g=1+np.sum((h-p[:4])**2*np.exp(p[4:8]),axis=1)
    elif typ=='metric':g=1+(hl@p[:5])**2+(hl@p[5:])**2
    elif typ=='quadmetric':g=1+np.exp(np.clip(hq@p,-8,8))
    elif typ=='norm':g=1+np.sum(h*h*np.exp(p),axis=1)
    elif typ=='orthmix':
     a=h+p[0]*np.sin(p[1]*h); return np.c_[L if form==0 else aff(a),quad(a)]
    else:
     u=(hl if form==0 else hq)@p;return np.c_[X,{'sinindex':np.sin,'tanhindex':np.tanh,'rtindex':warp['rat']}[typ](u)]
    return X/g[:,None] if form==0 else np.c_[L,X[:,5:]/g[:,None]]
   dim={'rad':8,'metric':10,'quadmetric':15,'norm':4,'orthmix':2}.get(typ,5 if form==0 else 15);rng=np.random.default_rng(14); P=[rng.normal(0,a,dim) for a in [.15,.5,1] for j in range(6)];P=[[-.4,2.9],[.4,2.15],[.2,4.]] if typ=='orthmix' else P
   for reg in [.01,.12]:
    def err(p):
     a=ds(p);e=a[I]@ls(a[I],Y[I,:2])-Y[I,:2];return np.r_[e.ravel(),reg*p]
    r=min((least_squares(err,p,max_nfev=75) for p in P),key=lambda r:r.cost);a=ds(r.x);c=ls(a[I],Y[I]);e=a@c-Y;h0=np.sum(np.linalg.qr(a[I])[0]**2,1);non.append((abs(e[I]/np.maximum(1-h0[:,None],.001)).mean(0),abs(e[J]).mean(0),kn,typ,form,reg,r.x));
for s0 in [0,1]:
 print('LINEAR',s0,*[(k,e[0],e[1],e[2]) for k,e,et in sorted(results,key=lambda r:r[1][s0][0])[:13]],sep='\n');print('NONLINEAR',s0,*sorted(non,key=lambda r:r[s0][0])[:9],sep='\n');
for top in [3,10,25,60]:
 m=sorted(results,key=lambda r:r[1][0][0])[:top];print('ensemble',top,abs(np.mean([r[1][3][J,0] for r in m],0)-y[J]).mean());
