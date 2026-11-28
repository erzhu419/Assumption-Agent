exec(open('/tmp/compose.py').read().split('for j in [')[0]); from scipy.special import erf
from scipy.optimize import minimize_scalar
F={'asinh':np.arcsinh,'atan':np.arctan,'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'cbrt':lambda x:np.cbrt(1+x*x),'rat':lambda x:x/(1+x*x),'sat':lambda x:x/np.sqrt(1+x*x),'evenrat':lambda x:1/(1+x*x),'erf':erf,'exp':lambda x:np.exp(-x*x)}
P={}
for k in ['v1','v2']:
 scores=[]; xx=c['u']; yy=c[k]
 for name,f in F.items():
  for lin in [False,True]:
   def d(x,g):return ab(np.c_[x,f(g*x)] if lin else f(g*x))
   def loss(g):
    a=d(xx,g); return np.mean((a@np.linalg.lstsq(a,yy,rcond=1e-9)[0]-yy)**2)
   gg=np.linspace(.15,5,50); j=np.argmin([loss(g) for g in gg]); g=minimize_scalar(loss,bounds=(gg[max(j-1,0)],gg[min(j+1,49)]),method='bounded').x
   scores.append((loss(g),name,lin,g,np.linalg.lstsq(d(xx,g),yy,rcond=1e-9)[0]))
 scores.sort(key=lambda x:x[0]); print('C panel',k,[(e,n,l,g) for e,n,l,g,_ in scores[:5]]); _,name,lin,g,q=scores[0]; P[k]=(name,lin,g,q)
 print('q',q.round(4))
def calc(k,x):
 name,lin,g,q=P[k]; a=F[name](g*x); return ab(np.c_[x,a] if lin else a)@q
x=U[:72]; z=V['b']; X={'u':x,'b':z,'cosu':np.cos(2.6*x)}
for k in P:
 for a,s0 in list(X.items())[:3]:X[k+a]=calc(k,s0)
S=[]
for a,x in X.items():
 S += [(a,x),(a+'q',np.c_[x,Q(x)])]
 for name,f in F.items():
  for g in (.22,.6,.78,1.,1.4,1.85,2.15,2.6,3.25,4.2,5.5):
   u=f(g*x); S += [(f'{a}-{name}{g}',u),(f'{a}+{name}{g}',np.c_[x,u]),(f'{a}-{name}{g}q',np.c_[u,Q(u)])]
R=[]
for name,a in S:
 for k,x in [('none',np.empty((72,0))),*X.items(),('c-b',np.c_[X['v1b'],X['v2b']]),('c-u',np.c_[X['v1u'],X['v2u']])]:
  d=ab(np.c_[a,x]); q=np.linalg.lstsq(d[idx],Y[idx],rcond=1e-8)[0]; r=abs(d@q-Y); R.append((r[~idx].mean(),r[idx].mean(),name,k))
print('CV',sorted(R)[:22]); print('train',sorted(R,key=lambda x:x[1])[:14]);
