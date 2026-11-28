import numpy as np,pickle,itertools
from scipy.optimize import least_squares,root,minimize_scalar
from scipy.linalg import svd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; F=lambda z:np.c_[np.ones(len(z)),z]
X=np.r_[s['train']['x'],s['test']['x']]; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; k,h1,h2,h3=s['telemetry_vec']; rng=np.random.default_rng(8)
u,v=[s['panel_a'][l] for l in ('u','v')]; ok=np.max(abs(u-v),1)<2; u,v=u[ok],v[ok]
ua=np.roll(u,-1,1); C=np.column_stack([L(F(np.c_[u[:,j],ua[:,j],u[:,j]*ua[:,j],np.sin(h1*u[:,j])]),v[:,j]) for j in range(4)])
def calib(q):return C[0]+C[1]*q+C[2]*np.roll(q,-1)+C[3]*q*np.roll(q,-1)+C[4]*np.sin(h1*q)
xx=[]
for x in X:
 guesses=[x,np.zeros(4),x*.8,x*1.3]+list(u[np.argsort(np.linalg.norm(v-x,axis=1))][:8])+list(rng.uniform(-1.5,1.5,(8,4))); rr=[root(lambda q:calib(q)-x,g).x for g in guesses]; q=min(rr,key=lambda q:np.linalg.norm(q) if np.linalg.norm(calib(q)-x)<1e-7 else 1e3+np.linalg.norm(calib(q)-x)); xx.append(q)
x=np.array(xx); print('root',np.max(abs(np.array([calib(q) for q in x])-X)),u.min(0),u.max(0),x.min(0),x.max(0))
ops={}
for key,op,p in [('b','v','panel_b'),('c','v1','panel_c'),('d','v2','panel_c')]:
 U,V=s[p]['u'],s[p][op]; out=[]
 for name,h0 in [('tanh',lambda z,a:np.tanh(a*z)),('sin',lambda z,a:np.sin(a*z)),('cos',lambda z,a:np.cos(a*z)),('soft',lambda z,a:z/(1+a*abs(z))),('rat',lambda z,a:z/(1+a*z*z)),('sqrt',lambda z,a:z/np.sqrt(1+a*z*z)),('norm',lambda z,a:z/(1+a*np.sum(z*z,1,keepdims=True))),('next',lambda z,a:z/(1+a*np.roll(z,-1,axis=1)**2))]:
  for base in [0,1]:
   f=lambda z,a,h=h0,base=base:F(np.c_[z,h(z,a)] if base else h(z,a)); loss=lambda a:np.linalg.norm(V-f(U,a)@L(f(U,a),V)); aa=minimize_scalar(loss,bounds=(.001,4),method='bounded').x; bb=L(f(U,aa),V); out.append((loss(aa),name,base,aa,bb,f))
 out.sort(key=lambda a:a[0]); print(key,[(round(l,6),n,i,round(a,5)) for l,n,i,a,_,_ in out[:7]]); _,_,_,a,b,f=out[0]; ops[key]=lambda z,a=a,b=b,f=f:f(z,a)@b
# stable shared temporal modes
basis=lambda r:np.array([t,-np.expm1(-r[0]*t),-np.expm1(-r[1]*t)])
r=least_squares(lambda r:(T[:52]-L(basis(r).T,T[:52].T).T@basis(r)).ravel(),[.8,7],bounds=([.1,2],[2,20])).x
B=basis(r); A=L(B.T,T.T).T; Y=np.c_[y,T,A]; print('rates',r,'amp sd',A.std(0))
def score(z,Y=Y):
 p=z@L(z[:52],Y[:52]); q=np.mean(abs(p[52:72]-Y[52:]),0); return np.round(np.r_[q[0],q[1:9].mean(),q[9:]],5)
D={'x':x};D.update({n:f(x) for n,f in ops.items()}); D.update({n+m:f(z) for n,z in list(D.items())[1:] for m,f in ops.items()});
quad=lambda x: np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
res=[]
for n,z in D.items():
 for typ,zz in [('lin',z),('quad',np.c_[z,quad(z)]),('trig',np.c_[z,np.sin(h1*z),np.cos(h1*z)]),('tanh',np.c_[z,np.tanh(h1*z)]),('rational',np.c_[z,z/(1+k*z*z),quad(z)])]:res.append((score(F(zz)),n,typ))
for i in range(5):print(i, sorted(res,key=lambda z:z[0][i])[:4])
np.savez('/tmp/coords2.npz',**D,Y=Y,B=B,X=X)
