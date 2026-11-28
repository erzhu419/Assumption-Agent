import numpy as np,pickle,importlib.util
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; n=len(X)
p='/tmp/hegel_continual_archive/conf_m102_a01_batch_03_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
A=lambda x: np.column_stack([np.ones(len(x)),x]); mae=lambda x: np.mean(abs(x),axis=0)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
from scipy.optimize import minimize_scalar
from itertools import combinations
p=S["panel_c"]; u=p["u"]; V=p["v1"]; W=p["v2"]
z,h=old._coordinates(S,X); p=S['panel_b']; B=np.linalg.lstsq(A(p['u']),p['v'],rcond=None)[0]; C={'x':X,'z':z,'bz':A(z)@B,'bx':A(X)@B,'h':h,'ih':np.linalg.solve(B[1:].T,(X-B[0]).T).T}; p=S['panel_c']; bv=np.linalg.lstsq(A(np.tanh(p['u'])),V,rcond=None)[0]; bw=np.linalg.lstsq(A(p['u']+.45*p['u']**2),W,rcond=None)[0]
for k,g in list(C.items()): C['v'+k]=A(np.tanh(g))@bv; C['w'+k]=A(g+.45*g*g)@bw
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
fa=lambda g: np.arctan(2.15*g); fb=lambda g:g+.446*g*g
ca=np.linalg.lstsq(A(fa(u)),V,rcond=None)[0]; cb=np.linalg.lstsq(A(fb(u)),W,rcond=None)[0]
base={k:C[k] for k in ['x','z','h','bx','bz','ih']}
for k,g in list(base.items()):
 for kk,gg in [('c1',A(fa(g))@ca),('c2',A(fb(g))@cb)]:
  base[k+kk]=gg
  if k in ['x','z']: base[k+kk+'b']=A(gg)@B
F=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(28).permutation(72),4)]
t=S['eval_times']; s=-np.expm1(-.8*t); y=np.c_[Y,T@s/(s@s)]
std=lambda a: a/np.maximum(np.sqrt(np.mean(a*a,axis=0)),1e-12)
quad=lambda g:np.column_stack([g[:,i]*g[:,j] for i,j in combinations_with_replacement(range(4),2)])
stack=lambda a,n:np.column_stack([np.prod(a[:,j],axis=1) for j in combinations_with_replacement(range(4),n)])
kern=lambda a:std(a)@std(a).T
D0=kern(A(X)); Dq=kern(quad(X)); Dc=kern(stack(X,3)); I=np.eye(len(X))
def run(ks):
 K=np.array(ks); out=[]; weights=[]
 for a,b in F:
  P=K[:,a][:,:,a]; Q=K[:,b][:,:,a]; pp=[]; ww=[]
  for yg in y[a].T:
   def obj(p):
    w=np.exp(p); c=cho_factor(np.einsum('i,ijk->jk',w,P),check_finite=False); alpha=cho_solve(c,yg,check_finite=False); inv=cho_solve(c,np.eye(len(a)),check_finite=False); grad=.5*np.einsum('jk,ijk->i',inv-np.outer(alpha,alpha),P)*w; return .5*yg@alpha+np.log(np.diag(c[0])).sum(),grad
   opt=minimize(obj,np.full(len(P),-4.),jac=True,bounds=[(-14,4)]*len(P),method='L-BFGS-B',options={'maxiter':80}); w=np.exp(opt.x); ww.append(w); pp.append(np.einsum('i,ijk->jk',w,Q)@np.linalg.solve(np.einsum('i,ijk->jk',w,P),yg))
  out.append(np.mean(abs(np.array(pp).T-y[b]),axis=0)); weights.append(ww)
 return np.array(out),np.array(weights)
rows=[]
for name,g in base.items():
 for fn,f in [('sin',np.sin(.7*g)),('tanh',np.tanh(1.25*g)),('rat',g/(1+.22*g*g)),('mixed',np.c_[np.sin(.7*g),np.tanh(1.25*g),g/(1+.22*g*g)])]:
  if fn=='mixed': ks=[D0,Dq,Dc,kern(f),kern(quad(g)),I]
  else: ks=[D0,Dq,Dc,kern(f),kern(quad(f)),kern(g*f),I]
  e,w=run(ks); rows.append((np.r_[e[0],e[1:].mean(0),e[1:].max(0)],name+':'+fn,w[0]))
for idx in range(4):
 print('RESULT',idx)
 for e,k,w in sorted(rows,key=lambda z:z[0][idx])[:6]:print(k,e,w)
