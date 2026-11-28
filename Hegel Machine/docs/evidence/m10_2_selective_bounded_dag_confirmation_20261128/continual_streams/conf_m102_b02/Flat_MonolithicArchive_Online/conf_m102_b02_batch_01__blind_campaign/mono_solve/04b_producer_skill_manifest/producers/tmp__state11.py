exec(open('/tmp/cv_exact.py').read().split('P=[np.arange')[0])
from scipy.optimize import least_squares
from scipy.special import erf
X=tr['x']; T=tr['traj']; tt=s['eval_times']; cal=m._calibration(s['panel_a']); z=np.linalg.solve(cal[1:].T,(X-cal[0]).T).T
pb,pc=s['panel_b'],s['panel_c']; B=m._operator(pb['u'],pb['v'],erf); C=m._operator(pc['u'],pc['v1'],np.arcsinh)
ls=m._ls
vdes=lambda w:np.c_[np.ones(len(w)),w,w*w]; vc=ls(vdes(pc['u']),pc['v2']); V=lambda w:vdes(w)@vc
b=B(z); h=C(b); r=C(z); v=V(b); zg=V(z); dot=lambda w:w.sum(1)[:,None]
base=[z,h,dot(z*h)]; baseline=np.c_[z,r,h,z*r,z*h,r*h]; basic=np.column_stack(base)
Ws={'z':z,'B':b,'C':r,'CB':h,'VB':v,'V':zg,'VC':V(r),'BC':B(r),'delta':z-h,'sum':z+h,'zCB':z*h,'zB':z*b}
Ps=[np.arange(72)]+[rng.permutation(72) for _ in range(12)]+[np.argsort(k*X[:,j]) for j in range(4) for k in [-1,1]]
def cv(D,alpha=.003):
 E=[]
 for p in Ps:
  i,j=p[:52],p[52:]; u=np.r_[i,j]; f=m._ridge(D[u],T[i],alpha,52); e=abs(f-T[j]); E.append([e.mean(),e[:,-1].mean()])
 E=np.array(E); return np.r_[E[0],E[:13].max(0),E[13:].max(0),E[13:].mean(0)]
print('base',cv(baseline)); print('basic',cv(basic))
Fs={**Ws}
for k,w in Ws.items():
 for label, f in [('sq',lambda x:x*x),('cube',lambda x:x**3),('rat',lambda x:x/(1+x*x)),('abs',np.abs),('log',lambda x:np.log1p(x*x))]+[(f'{name}{a}',lambda x,fn=fn,a=a:fn(a*x)) for name,fn in [('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh)] for a in [.6,1.25,2.5,3.5,5]]:
  Fs[k+label]=f(w)
for k,w in list(Ws.items())[:8]:
 for l,g in list(Ws.items())[:8]:
  if k<l: Fs[k+'x'+l]=w*g
rows=[]
for name,D in Fs.items():
 for label,d in [('full',D),('sum',dot(D))]:
  for typ,bs in [('s',basic),('l',baseline)]:
   f=np.c_[bs,d]; e=cv(f); rows.append((e[4]+.35*e[2]+.5*e[6],name,label,typ,e))
for a in sorted(rows,key=lambda t:t[0])[:22]:print(a)
# Look at transferable residual directions and their dependence on individual branches.
F=lambda q:-np.expm1(-tt[:,None]*np.exp(q))
o=least_squares(lambda q:(F(q)@ls(F(q),T.T)-T.T).ravel(),np.log([.41,1.795])); H=ls(F(o.x),T.T).T
print('time',np.exp(o.x),'basic coefs',ls(np.c_[np.ones(72),basic],H));
