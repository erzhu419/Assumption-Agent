from probe import *
from scipy.linalg import polar
from scipy.optimize import minimize_scalar
from itertools import product
p=s['panel_c']; u=p['u'];
def panel(f,y,lo,hi):
 def F(w,a):return np.c_[np.ones(len(w)),w,f(w,a)]
 r=minimize_scalar(lambda a: np.linalg.norm(F(u,a)@fit(F(u,a),y)-y),bounds=(lo,hi),method='bounded'); B=fit(F(u,r.x),y); print('panel',r.x,np.std(F(u,r.x)@B-y));return lambda w:F(w,r.x)@B, polar(B[1:5])[0]
c,_=panel(lambda w,a:np.sign(w)*np.log1p(a*abs(w)),p['v1'],.5,3)
d,_=panel(lambda w,a:w*w/(1+a*w*w),p['v2'],0,1)
p=s['panel_a']; U=poly(p['u'],1); Y=p['v']; B=fit(U,Y)
for _ in range(6):
 k=np.argsort(np.linalg.norm(U@B-Y,axis=1))[:-3]; B=fit(U[k],Y[k]);
a,g=B[0],np.diag(B[1:]); p=s['panel_b']; B=fit(poly(p['u'],1),p['v']);
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; I=np.arange(len(x)); folds=[I[56:]]+list(np.array_split(np.random.default_rng(42).permutation(len(I)),4))+[np.argsort(np.linalg.norm(x,axis=1))[-12:]]
A={'x':x,'a':x*g+a,'i':(x-a)/g}; W=dict(A)
for n,w in A.items(): W[n+'b']=w@B[1:]+B[0]; W[n+'j']=(w-B[0])@np.linalg.inv(B[1:]);
F0=poly(x,2); F1=poly(x,1); Q0=np.linalg.qr(F0)[0]; R=[]
def eval(F,tags):
 G=F.copy(); G-=Q0@(Q0.T@G); sd=np.std(G,axis=0); G=G[:,sd>1e-8]/sd[sd>1e-8]; K=G@G.T; K0=10000*F0@F0.T
 for l in (.03,.3,3,30):
  E=[]
  for v in folds:
   t=np.setdiff1d(I,v); M=K+K0; pred=M[np.ix_(v,t)]@np.linalg.solve(M[np.ix_(t,t)]+l*np.eye(len(t)),Y[t]); E.append(np.mean(abs(pred-Y[v]),axis=0))
  E=np.array(E); R.append((np.array([E[:5,0].mean(),E[:5,1:].mean(),E[:5,0].max(),E[:5,1:].max(),E[-1,0],E[-1,1:].max()]),tags,l))
for n,w in W.items():
 C=c(w); D=d(w); P=poly(w,3)[:,15:];
 def inter(U,V):return (U[:,:,None]*V[:,None,:]).reshape(len(U),-1)
 families={'c':C,'d':D,'cd':np.c_[C,D],'c*d':inter(C,D),'w*c':inter(w,C),'w*d':inter(w,D),'c2':poly(C,2),'d2':poly(D,2),'cd2':np.c_[poly(C,2),poly(D,2),inter(C,D)],'all':np.c_[C,D,inter(w,C),inter(w,D),inter(C,D),P]}
 for k,F in families.items():eval(F,(n,k))
for h,m in [(0,4),(1,5)]:print('head',h,*sorted(R,key=lambda o:o[0][h]+.25*o[0][m])[:9],sep='\n')
