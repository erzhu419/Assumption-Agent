exec(open('/tmp/scan.py').read().split("p=s['panel_a'];")[0])
from scipy.optimize import root
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.argsort(np.linalg.norm(design(u)@ls(design(u),v)-v,axis=1))[:45]; u,v=u[keep],v[keep]
ijk=[i for d in range(1,4) for i in itertools.combinations_with_replacement(range(4),d)]
g=design(np.column_stack([np.prod(u[:,i],axis=1) for i in ijk])); coef=ls(g,v)
for j in range(4): print('A cubic',j,coef[0,j],[(i,round(w,5)) for i,w in zip(ijk,coef[1:,j]) if abs(w)>.005])
terms=[]
for j in range(4):
 ii=[i for i,w in zip(ijk,coef[1:,j]) if abs(w)>.005]; terms.append((ii,ls(design(np.column_stack([np.prod(u[:,i],axis=1) for i in ii])),v[:,j])))
def A(x):return np.column_stack([design(np.column_stack([np.prod(x[:,i],axis=1) for i in ii]))@w for ii,w in terms])
print('A sparse residual',np.sqrt(np.mean((A(u)-v)**2)))
def Ai(x):
 start=design(x)@ls(design(v),u); z=[]; err=[]
 for xi,zi in zip(x,start):
  fit=root(lambda a:A(a[None,:])[0]-xi,zi); z.append(fit.x); err.append(np.linalg.norm(fit.fun))
 print('inverse max residual',max(err));return np.array(z)
p=s['panel_b'];B=fit(p['u'],p['v'],design);Bi=fit(p['v'],p['u'],design)
p=s['panel_c'];C=fit(p['u'],p['v1'],lambda x: design(np.sign(x)*(-np.expm1(-1.8*abs(x)))));D=fit(p['u'],p['v2'],lambda x:design(x,x*x/np.sqrt(1+.389*x*x)))
x=s['train']['x'];t=s['train']['traj'];y=s['train']['y'];T=np.c_[y,t]
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def quad(a): i,j=np.triu_indices(a.shape[1]);return a[:,i]*a[:,j]
def score(g, ridge):
 h=g[:52];d=np.linalg.norm(h,axis=0);d=np.where(d>1e-10,d,1);g=g/d; h=g[:52]; w=np.linalg.solve(h.T@h+np.diag(np.r_[1e-10,np.full(h.shape[1]-1,ridge)]),h.T@T[:52]); r=abs(g[52:]@w-T[52:]);return r[:,0].mean(),r[:,1:].mean()
roots={'x':x,'A':A(x),'Ai':Ai(x),'B':B(x),'Bi':Bi(x),'BA':B(A(x)),'BAi':B(Ai(x)),'BiA':Bi(A(x)),'BiAi':Bi(Ai(x)),'AB':A(B(x)),'AiB':Ai(B(x))}; out=[]
for nm,a in roots.items():
 c,d=C(a),D(a)
 opts={'lin':design(x),'poly':design(x,quad(a)), 'c':design(x,c),'d':design(x,d),'cd':design(x,c,d),'qc':design(x,c,quad(c)), 'qd':design(x,d,quad(d)), 'cdX':design(x,c,d,cross(c,d)),'cX':design(x,c,cross(a,c)), 'dX':design(x,d,cross(a,d)), 'cdQ':design(x,c,d,quad(a),quad(c),quad(d)), 'all':design(x,c,d,quad(a),quad(c),quad(d),cross(a,c),cross(a,d),cross(c,d)),'cD':design(x,c,D(c),cross(c,D(c))), 'dC':design(x,d,C(d),cross(d,C(d)))}
 for name,g in opts.items():
  for r in (1e-7,1e-4,.003,.03):out.append((*score(g,r),nm,name,r))
for idx in (0,1,2):
 print('leaders',idx);print('\n'.join(str(a) for a in sorted(out,key=lambda v:v[idx] if idx<2 else max(v[:2]))[:18]))
