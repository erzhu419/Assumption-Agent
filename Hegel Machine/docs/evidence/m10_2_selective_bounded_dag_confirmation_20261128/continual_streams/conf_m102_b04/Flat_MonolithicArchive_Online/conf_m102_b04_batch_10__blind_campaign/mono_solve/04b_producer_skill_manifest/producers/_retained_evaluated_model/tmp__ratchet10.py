exec(open('/tmp/cv10.py').read().split("print('official'")[0])
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
z=m._calibration(s['panel_a'])(x); p=s['panel_b']; u0,v=p['u'],p['v']; A=m._design; fit=m._fit; y=u['y']; c=m._operator(s['panel_c'],'v1',lambda x,k:np.arcsinh(k*x),(.05,6.))(z)
ops={};
for name,g,p0,lo,hi in [('sin',lambda x,p:np.sin(p[0]*x)+p[1]*x,[2.8,-2.5],[.5,-10],[5,10]),('rat',lambda x,p:x/(1+p[0]*x*x)+p[1]*x,[.75,-.94],[.01,-5],[5,5]),('tanh',lambda x,p:np.tanh(p[0]*x)+p[1]*x,[1.4,-1.3],[.05,-10],[5,10]),('rat3',lambda x,p:x**3/(1+p[0]*x*x),[.75],[.01],[8]),('cub',lambda x,p:x**3+p[0]*x,[.1],[-5],[5]),('rat2',lambda x,p:x*(1+p[1]*x*x)/(1+p[0]*x*x+p[2]*x**4),[.75,-10,.05],[.01,-100,0],[8,10,8])]:
 def res(p,coef=False):
  d=A(g(u0,p)); b=fit(d,v); return b if coef else (d@b-v).ravel()
 r=least_squares(res,p0,bounds=(lo,hi),max_nfev=400); ops[name]=A(g(z,r.x))@res(r.x,True); print(name,np.round(r.x,5),np.sqrt(np.mean(r.fun**2)))
for name,g in [('R',lambda x,k:x/(1+k*x*x)),('S',lambda x,k:np.sin(k*x))]: ops[name]=m._operator(p,'v',g,(.05,4.))(z)
rng=np.random.default_rng(5012); rand=[rng.permutation(idx) for _ in range(25)]; splits=[(idx[:52],idx[52:])]+[(i[:52],i[52:]) for i in rand]; tails=[]
for w in [*x.T,np.linalg.norm(x,axis=1)]:
 j=np.argsort(w); tails.extend([(j[:52],j[52:]),(j[20:],j[:20])]);
quad=lambda z:np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)])
res=[]
for kn,b in ops.items():
 for an,a in [('x',x),('z',z),('c',c),('b',b)]:
  for qn,q in [('none',None),('square',a*a),('pair',quad(a)),('cross',(b[:,:,None]*a[:,None,:]).reshape(-1,16)),('bpair',quad(b)),('prodopp',a*np.roll(a,2,axis=1)),('abs',abs(a)),('sin',np.sin(2*a)),('tanh',np.tanh(a)),('three',a**3),('mixed',np.column_stack((quad(a),quad(b))))]:
   f=A(b,a,*([q] if q is not None else []))
   for l in [.0001,.003,.02,.08,.25]:
    E=np.array([abs(f[j]@fit(f[i],y[i],l)-y[j]).mean() for i,j in splits+tails]); res.append((max(E),E[0],E[1:26].max(),E[26:].max(),E[26:].argmax(),kn,an,qn,l,E[1:26].mean()))
print('baseline',next(r for r in res if r[5:9]==('R','x','pair',.0001)))
for i in sorted(res)[:20]: print(i)
print('best means');
for i in sorted(res,key=lambda r:r[-1])[:12]: print(i)
