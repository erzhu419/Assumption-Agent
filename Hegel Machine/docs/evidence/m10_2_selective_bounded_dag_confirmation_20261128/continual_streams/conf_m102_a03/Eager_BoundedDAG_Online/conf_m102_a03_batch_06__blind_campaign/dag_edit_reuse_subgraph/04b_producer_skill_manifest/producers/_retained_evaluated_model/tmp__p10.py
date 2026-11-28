import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
from scipy.optimize import minimize_scalar,least_squares
from scipy.spatial.distance import cdist
from scipy.special import erf
from scipy.linalg import solve
print('frozen align source:'); import inspect;print(inspect.getsource(f.run_node))
W={};Y=np.c_[y,T]; n=52; rng=np.random.default_rng(211); folds=[(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(rng.permutation(n),4)];folds.append((np.arange(n),np.arange(n,len(y))))
def measure(pred):
 m=np.array([[np.mean(abs(E[:,0])),np.mean(abs(E[:,1:]))] for E in pred]);return np.r_[m[:-1].mean(0),m[-1]]
def ev(D):return measure([D[b]@ls(D[a],Y[a])-Y[b] for a,b in folds])
def app(x,h,typ):return {'gau':lambda:x*np.exp(-(h*x)**2),'sin':lambda:np.sin(h*x),'atan':lambda:np.arctan(h*x),'cos':lambda:1/(1+(h*x)**2),'tanh':lambda:np.tanh(h*x),'erf':lambda:erf(h*x)}[typ]()
def mapping(p,k,types):
 U=s[p]['u'];V=s[p][k]; cand=[]
 for typ in types:
  def design(x,h):return np.c_[f._affine(x),app(x,h,typ)]
  def err(h):
   D=design(U,h);return np.linalg.norm(D@ls(D,V)-V)
  g=np.linspace(.2,4,40); j=np.argmin([err(h) for h in g]); q=minimize_scalar(err,bounds=(max(.05,g[j]-.15),g[j]+.15),method='bounded'); C=ls(design(U,q.x),V);cand.append((q.fun,typ,q.x,C))
 e,typ,h,C=min(cand,key=lambda a:a[0]); print('map',p,k,e,typ,h,'panel rms',np.std(U,axis=0));return lambda x: np.c_[f._affine(x),app(x,h,typ)]@C
B=mapping('panel_b','v',['gau','sin']); C=mapping('panel_c','v1',['gau','sin','atan','tanh','erf','cos']); F=mapping('panel_c','v2',['cos']); X=B(raw);
print('raw invert mismatch',np.max(abs(raw-f._inverse(tr['x'],f._fit_sensor(s['panel_a'])))));print('panel ranges and Z',[(k,np.min(s[k]['u']),np.max(s[k]['u'])) for k in ('panel_a','panel_b','panel_c')],X.min(),X.max());
for nm,U in [('B',X),('raw',raw),('C(B)',C(X)),('F(B)',F(X)),('C(raw)',C(raw)),('F(raw)',F(raw)),('B(C(raw))',B(C(raw))),('B(F(raw))',B(F(raw)))]:
 Q=poly(U,2)[0];print(nm,'Q',ev(Q).round(4));print(nm,'Q+affC+affF',ev(np.c_[Q,C(U),F(U)]).round(4));print(nm,'Q+quadC',ev(np.c_[Q,poly(C(U),2)[0][:,1:]]).round(4));
 if nm not in ('B','raw','C(B)','F(B)'):continue
 R=[]
 for typ in ('cos','sin','gau','atan','tanh'):
  for h in (.4,.65,1.,1.3,1.6,2.2,2.8,3.5):
   A=app(U,h,typ);P=lambda x:poly(x,2)[0]
   for j,D in enumerate([np.c_[Q,A],np.c_[Q,A,U*A],np.c_[Q,A,U*A,np.stack([U[:,i]*A[:,j]+U[:,j]*A[:,i] for i in range(4) for j in range(i+1,4)],1)],P(A),np.c_[P(A),U],np.c_[Q,P(A)[:,1:]]]):
    E=ev(D);R.append((E[:2].mean(),typ,h,j,E))
 print('best feat',nm,[(typ,h,j,E.round(4)) for _,typ,h,j,E in sorted(R,key=lambda a:a[0])[:8]])
 if nm not in ('B','raw'):continue
 R=[]; A=f._affine(U); D2=cdist(U,U,'sqeuclidean');
 for typ in ('gauss','matern','laplace','poly3','poly4','poly5','poly6'):
  for h in (.2,.4,.7,1.,1.5,2.,3.,5.,8.):
   r=np.sqrt(D2)*h;G={'gauss':lambda:np.exp(-r*r/2),'matern':lambda:(1+r+r*r/3)*np.exp(-r),'laplace':lambda:np.exp(-r)}.get(typ,lambda:(1+h*U@U.T)**int(typ[-1]))();G/=np.mean(np.diag(G));
   for a0 in (.00001,.0001,.001,.01,.1):
    for D in (A,Q):
     E=[]
     for a,b in folds:
      K=G[np.ix_(a,a)]+a0*np.eye(len(a)); cross=G[np.ix_(b,a)]; aD=D[a]; sol=solve(K,np.c_[Y[a],aD],assume_a='pos'); S=sol[:,:Y.shape[1]]; F0=sol[:,Y.shape[1]:]; co=ls(aD.T@F0,aD.T@S); E.append(cross@(S-F0@co)+D[b]@co-Y[b]);
     M=measure(E);R.append((M[:2].mean(),typ,h,a0,D.shape[1],M))
 print('kernel',nm,[(typ,h,a,d,E.round(4)) for _,typ,h,a,d,E in sorted(R,key=lambda a:a[0])[:8]])
# Assess how closely readouts agree with a common squashed affine coordinate and its quadratic interactions.
for xx,nm in [(X,'B'),(raw,'raw'),(C(X),'CB')]:
 for typ in ['rat','sin','tanh','gau','poly']:
  for mode in [0,1,2]:
   def F0(x,p):
    h=np.exp(np.clip(p[0],-4,2)); z=x if mode==0 else x+p[3:7] if mode==1 else x@p[3:].reshape(4,4);
    W={'rat':lambda:z/(1+(h*z)**2),'sin':lambda:np.sin(h*z),'tanh':lambda:np.tanh(h*z),'gau':lambda:z*np.exp(-(h*z)**2),'poly':lambda:z**3}[typ]();return x+p[1]*W+p[2]*z**2
   def feat(x,p):return poly(F0(x,p),2)[0]
   out=[]; pars=[]
   for a,b in (folds[0],folds[-1]):
    def res(p):
     D=feat(xx[a],p);return (D@ls(D,Y[a])-Y[a]).ravel()
    pextra=[] if mode==0 else [0]*4 if mode==1 else np.eye(4).ravel(); opt=min([least_squares(res,np.r_[h,b0,0,pextra],max_nfev=55) for h,b0 in [(-.7,.3),(-.2,-.3),(.6,.3)]],key=lambda q:q.cost); DD=feat(xx,opt.x); E=abs(DD[b]@ls(DD[a],Y[a])-Y[b]);out.append((E[:,0].mean(),E[:,1:].mean()));pars.append(opt.x[:3].round(3));
   print('super',nm,typ,mode,np.array(out).ravel().round(4),pars)
