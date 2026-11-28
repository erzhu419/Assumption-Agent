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
print('B errors',np.sqrt(np.mean((B(b['u'])-b['v'])**2,0)));print('C2 errors',np.sqrt(np.mean((C2(p['u'])-p['v2'])**2,0)))
print('B',cb);print('C1',c1);print('C2',lin(one(np.cos(2.4*p['u'])),p['v2']));
a=np.arange(52);h=np.arange(52,72); folds=[(a,h)]+[(ix[:52],ix[52:]) for ix in [np.random.default_rng(i).permutation(n) for i in range(4)]]
F=-np.expm1(-t[:,None]*np.array([.37633,1.78968,5.5352])); H=lin(F,T.T).T; L=np.c_[y,H,T]
D={'z':z,'b':B(z),'c':C(z),'d':C2(z),'cb':C(B(z)),'db':C2(B(z)),'p':mon(z,cyc),'q':np.cos(2.4*z),'qb':np.cos(2.4*B(z))};
for k,X in D.items(): print(k,np.round(np.corrcoef(np.c_[X,L[:,:4]].T)[:4,4:],2))
cols={};groups={}
def add(name,X):
 X=np.reshape(X,(n,-1)); groups[name]=X
 for i,g in enumerate(X.T):cols[name+':'+str(i)]=g
for name,X in D.items():
 add(name,X)
 for d in [2,3]:add(name+'P'+str(d),mon(X,list(itertools.combinations_with_replacement(range(4),d))))
 for k in [.5,1.25,2,2.4,3,4,5,6,8]:
  for fun in [np.sin,np.cos,np.tanh]:add(name+fun.__name__+str(k),fun(X*k))
 for k in [.22,.5,1,2]:
  add(name+'rat'+str(k),X/(1+k*X*X));add(name+'rat2'+str(k),1/(1+k*X*X))
for n1,n2 in [('z','c'),('z','d'),('c','d'),('b','c'),('b','d'),('b','cb'),('b','db'),('cb','db'),('z','b'),('z','cb'),('z','db'),('p','q')]:
 W,V=D[n1],D[n2];add(n1+'*'+n2,(W[:,:,None]*V[:,None,:]));add(n1+'/'+n2,(W[:,:,None]/(1+.22*V[:,None,:]**2)));add(n2+'/'+n1,(V[:,:,None]/(1+.22*W[:,None,:]**2)))
for k,X in D.items():
 for name,V in [('sum',X[:,[i for i,j in pairs]]+X[:,[j for i,j in pairs]]),('dif',X[:,[i for i,j in pairs]]-X[:,[j for i,j in pairs]])]:
  for f in [np.sin,np.cos,np.tanh]:
   for r in [1,1.25,2.4,3.5,5,6]:add(k+name+f.__name__+str(r),f(r*V))
from sklearn.linear_model import Lars
M=np.column_stack(list(cols.values())); N=np.array(list(cols)); print('M',M.shape)
for j in [0,1,2,3,4,11]:
 target=L[:,j]; print('TARGET',j); opt=[]
 for ks in [(k,) for k in D]+[('z','p','q','c','d'),('b','cb','db','qb'),('z','b','c','d'),tuple(D)]:
  mask=np.array([any(k in ks for k in name.split(':')[0].replace('*','/').split('/')) or any(name.startswith(k+p) for k in ks for p in ['P','sin','cos','tanh','rat','sum','dif']) for name in N]); X=M[:,mask]; names=N[mask]; E=[]; traces=[]
  for aa,hh in folds:
   m=X[aa].mean(0); sd=np.maximum(X[aa].std(0),1e-9); XX=(X-m)/sd; mdl=Lars(n_nonzero_coefs=28,fit_intercept=True).fit(XX[aa],target[aa]); pred=XX@mdl.coef_path_+mdl.intercept_; E.append(np.mean(abs(pred[hh]-target[hh,None]),0)); traces.append(mdl.active_)
  e=np.stack([np.pad(v,(0,29-len(v)),mode='edge') for v in E]); i=e.mean(0).argmin();opt.append((e.mean(0)[i],e[0,i],max(e[:,i]),str(ks),i,names[traces[0][:min(i,12)]]))
 for r in sorted(opt,key=lambda v:v[0])[:4]:print(r)
