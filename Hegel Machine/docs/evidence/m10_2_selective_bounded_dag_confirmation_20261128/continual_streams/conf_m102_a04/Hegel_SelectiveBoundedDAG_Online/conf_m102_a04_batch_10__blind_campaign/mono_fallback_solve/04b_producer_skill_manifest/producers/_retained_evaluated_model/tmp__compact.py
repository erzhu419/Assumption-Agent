exec(open('/tmp/stress.py').read().split('out=[]')[0])
c,d=m._operators(s['panel_c'],z); cd=(c[:,:,None]*d[:,None,:]).reshape(len(c),16); A=m._one(np.c_[c,d,np.sum(c*d,axis=1),c[:,0]*d[:,1]]); print('compact full coeff (y and trajectory time points)'); print(np.round(m._ls(A,np.c_[y,T]),4)); Y=np.c_[y,T];
for name,iy,it in [('dense',F,F),('compact',A,A),('decoupled',A[:,:-1],A),('ranktwo',A[:,:-1],A)]:
 out=[]
 for l,b in splits:
  a=np.setdiff1d(allidx,b); py=iy[b]@m._ls(iy[a],y[a]);
  if name=='ranktwo':
   q=m._time(s['eval_times'],T[a]); pt=it[b]@m._ls(it[a],m._ls(q,T[a].T).T)@q.T
  else:pt=it[b]@m._ls(it[a],T[a]);
  out.append((l,np.mean(abs(py-y[b])),np.mean(abs(pt-T[b])),np.max(np.mean(abs(pt-T[b]),0))))
 print(name,out[0]); print('worst random, tail',[(max(q,key=lambda t:t[j])[:j+1]) for q in (out[1:25],out[25:]) for j in range(1,4)])
p=s['panel_c']; u,V=p['u'],p['v1']; from scipy.optimize import minimize_scalar
for name,fun in [('asinhos',lambda a,q:np.arcsinh(q*a)),('atan',lambda a,q:np.arctan(q*a)),('tanh',lambda a,q:np.tanh(q*a)),('sin',lambda a,q:np.sin(q*a)),('sqrt',lambda a,q:a/np.sqrt(1+q*a*a)),('erf',lambda a,q:__import__('scipy').special.erf(q*a)),('rational',lambda a,q:a/(1+q*a*a))]:
 def basis(a,q,extra):return m._one(np.c_[a,fun(a,q),a**3] if extra else np.c_[a,fun(a,q)])
 for extra in (False,True):
  def err(q):
   B=basis(u,q,extra); return np.std(B@m._ls(B,V)-V)
  grid=np.linspace(.2,5,80); j=np.argmin([err(q) for q in grid]); r=minimize_scalar(err,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded'); print(name,extra,r.x,r.fun)
