import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; n=len(y)
np.set_printoptions(precision=4,suppress=True,linewidth=155)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(a):return np.c_[np.ones(len(a)),a]
def mon(a,idx):return np.column_stack([np.prod(a[:,j],axis=1) for j in idx])
pairs=list(itertools.combinations(range(4),2))
cyc=[(0,1),(1,2),(2,3),(3,0)]
def ca(a):return np.c_[one(a),mon(a,cyc),a**3]
A=s['panel_a']; u,v=A['u'],A['v']; keep=abs(v).max(1)<4; c=lin(ca(u[keep]),v[keep]); c[abs(c)<.001]=0
print('t/telemetry',t,s['telemetry_vec'],'calib',c)
def inv(x):
 return np.array([least_squares(lambda a:(ca(a[None])@c).ravel()-q,np.linalg.solve(c[1:5].T,q-c[0]),gtol=1e-11).x for q in x])
z=inv(tr['x']); b=s['panel_b']; cb=lin(one(b['u']),b['v']); B=lambda x:one(x)@cb
p=s['panel_c']; C1=lambda x:np.c_[np.ones(len(x)),mon(x,cyc)]; c1=lin(C1(p['u']),p['v1']); C=lambda x:C1(x)@c1

def c2fit(u,v):
 for name,fn in [('expquad',lambda a,k:np.exp(-k*a*a)),('cos',lambda a,k:np.cos(k*a)),('rational',lambda a,k:1/(1+k*a*a)),('tanh',lambda a,k:np.tanh(k*a))]:
  def f(a,k):return np.c_[one(a),fn(a,k)]
  opt=least_squares(lambda k:(f(u,k)@lin(f(u,k),v)-v).ravel(),[1],bounds=(.02,6)); k=opt.x; F=f(u,k); co=lin(F,v); er=np.sqrt(np.mean((F@co-v)**2)); print('C2 compact',name,k,er); yield er,lambda a,f=f,k=k,co=co:f(a,k)@co
C2=lambda a:one(np.cos(2.4*a))@lin(one(np.cos(2.4*p["u"])),p["v2"])
F=-np.expm1(-t[:,None]*np.array([.37633,1.78968,5.5352])); H=lin(F,T.T).T; L=np.c_[y,H]; a=np.arange(52); h=np.arange(52,72)
from scipy.linalg import qr
W=B(z); P=mon(z,cyc); Q=np.cos(2.4*z); R=np.cos(2.4*W); Cx=C(z); Dx=C2(z)
print('ranges',z.min(0),z.max(0),'inverse errors', np.max(abs(ca(z)@c-tr['x'])));print('rates fits baseline')
def jd(m):
 E=m@lin(m[a],L[a])-L;return np.round(np.mean(abs(E[a]),0),4),np.round(np.mean(abs(E[h]),0),4)
for k,m in [('z',one(z)),('z2',one(np.c_[z,mon(z,list(itertools.combinations_with_replacement(range(4),2)))])),('z3',one(np.c_[z,mon(z,list(itertools.combinations_with_replacement(range(4),2))),mon(z,list(itertools.combinations_with_replacement(range(4),3)))])),('zPQ',one(np.c_[z,P,Q])),('zPQR',one(np.c_[z,P,Q,R])),('known',one(np.c_[z,P,Q,R,mon(W,cyc),np.sin(1.25*W),np.sin(1.25*z)]))]: print(k,jd(m))
# Scan free Fourier ridges after removing low-complexity measured operators; jointly refine multiple peaks.
rng=np.random.default_rng(313); D=np.r_[rng.normal(size=(22000,4))*rng.uniform(.4,6,(22000,1)),cb[1:].T*1.25,cb[1:].T*2.4,np.eye(4)*2.4]
SW=np.sin(z@D.T); CW=np.cos(z@D.T)
base={ 'aff':one(z),'phys':one(np.c_[z,P,Q,R]),'quad':one(np.c_[z,mon(z,list(itertools.combinations_with_replacement(range(4),2))),Q,R]) }
for k,G in base.items():
 for j in range(4):
  best=[]; yy=L[:,j]; Ws=np.empty((0,4))
  for count in range(1,5):
   def design(w):
    P0=z@w.reshape(-1,4).T;return np.c_[G,np.sin(P0),np.cos(P0)]
   Q0=qr(design(Ws)[a],mode='economic')[0]; res=yy[a]-Q0@(Q0.T@yy[a]); s=SW[a]-Q0@(Q0.T@SW[a]); c0=CW[a]-Q0@(Q0.T@CW[a]); vs=(s*s).sum(0); vc=(c0*c0).sum(0); cs=(s*c0).sum(0); u=s.T@res; v=c0.T@res; scores=(u*u*vc+v*v*vs-2*u*v*cs)/np.maximum(vs*vc-cs*cs,1e-12); idx=np.argsort(scores)[-100:][::-1]; seeds=[]
   for ii in idx:
    if all(min(np.linalg.norm(D[ii]-vv),np.linalg.norm(D[ii]+vv))>.65 for vv in seeds):seeds.append(D[ii]);
    if len(seeds)>=5:break
   def fun(w):
    M=design(w)[a];return M@lin(M,yy[a])-yy[a]
   opt=min((least_squares(fun,np.r_[Ws.ravel(),w],max_nfev=85) for w in seeds),key=lambda o:o.cost); Ws=opt.x.reshape(-1,4); M=design(Ws); coef=lin(M[a],yy[a]); err=abs(M@coef-yy);best.append((np.round(err[a].mean(),4),np.round(err[h].mean(),4)))
   if err[h].mean()<.05 or count==2: print('ridge',k,j,count,best[-1],np.round(Ws,3),np.round(coef[-2*count:],3))
  print('ridge summary',k,j,best)
