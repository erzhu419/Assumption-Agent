from operators import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
from itertools import combinations
c=s['panel_c']; u=c['u']; print('frequency scan');
for g in [1.25,1.8,2.2,2.4,2.5,2.75,3.0,3.125,3.5]:
 D=F(np.cos(u*g)); print(g,np.linalg.norm(D@fit(D,c['v2'])-c['v2']))
x=s['train']['x']; z=(x-a)/b; t=s['eval_times']; H=-np.expm1(-t[:,None]*np.array([.38,1.8,5.4])); Y=np.c_[s['train']['y'],fit(H,s['train']['traj'].T).T]; n=52
P=PolynomialFeatures(3).fit_transform(z); Q=P[:,:15]; w=Bf(z); D=PolynomialFeatures(3).fit_transform(w)
def err(P,y):
 r=P@fit(P[:n],y[:n])-y;return abs(r[:n]).mean(),abs(r[n:]).mean()
R=[[] for i in range(4)]
def check(A,key):
 for j in range(4):R[j].append((*err(A,Y[:,j]),key))
for k,v in [('z',z),('B',w),('BT',z@B[1:].T),('zw',np.c_[z,w])]:
 Z=np.c_[v,*[v[:,i]+sg*v[:,j] for i,j in combinations(range(v.shape[1]),2) for sg in [-1,1]]];
 for i,d in enumerate(Z.T):
  for g in [.11,.22,.39,.5,1.,1.25,2.,3.]:
   den=1+g*d*d
   check(Q/den[:,None],(k,i,g,'q/'));check(P/den[:,None],(k,i,g,'p/'));check(np.c_[P,Q/den[:,None]],(k,i,g,'p+q/'))
   for nm,f in [('sin',np.sin),('cos',np.cos),('soft',lambda w:w/(1+abs(w))),('abs',np.abs),('tanh',np.tanh)]:
    v1=f(g*d); check(np.c_[Q,F(z)*v1[:,None]],(k,i,g,nm,'q+lin'));check(np.c_[P,v1],(k,i,g,nm,'p+'))
for i,r in enumerate(R):print('scan',i,sorted(r)[:3],sorted(r,key=lambda x:x[1])[:3]);
rng=np.random.default_rng(10)
for j in [0,1,2]:
 y=Y[:,j]; allres=[]
 for typ,func in [('cos',np.cos),('soft',lambda x:x/(1+abs(x))),('tanh',np.tanh),('rat',lambda x:1/(1+x*x))]:
  for nu in [1,2,3]:
   L=np.c_[z,np.ones(len(z))];
   for initial in range(16):
    def des(p):
     A=func(L@p.reshape(5,nu)); return np.c_[Q,A] if nu>1 else np.c_[Q,F(z)*A]
    def residual(p):
     A=des(p)[:n];return A@fit(A,y[:n])-y[:n]
    p=rng.normal(0,1,(5,nu)); q=least_squares(residual,p.ravel(),max_nfev=90); allres.append((*err(des(q.x),y),typ,nu,q.x.round(4)))
 print('optimized',j,sorted(allres,key=lambda r:r[0])[:4],sorted(allres,key=lambda r:r[1])[:4])
