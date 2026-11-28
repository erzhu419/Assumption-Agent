exec(open('/tmp/check_candidate.py').read().split('for i,p in enumerate(splits):')[0])
from scipy.optimize import minimize_scalar
from scipy.special import expit
z=m._calibrate(s['panel_a'],x); v=z*np.roll(g,1,axis=1); q=g[:,:2]*g[:,2:]; A=m._aff(np.c_[g,z,v,q]); ebase=np.c_[z*g,z*np.roll(v,2,axis=1),g*np.roll(g,1,axis=1)]
p=s['panel_c']; u=p['u']; test={}
for k,fn in [('v1',np.arctan),('v2',lambda w:w*expit(w))]:
 def d(a,w): return np.c_[m._aff(a),fn(a*w)]
 def loss(w):
  X=d(u,w); return np.mean((X@m._fit(X,p[k])-p[k])**2)
 w=minimize_scalar(loss,bounds=(.2,4),method='bounded').x; b=m._fit(d(u,w),p[k]);
 for k2,a in [('z',z),('g',g),('v',v),('g+.7v',g+.7*v)]:
  h=d(a,w)@b; test[k+k2]=h; test[k+k2+'cyclic']=h*np.roll(z,1,axis=1)
def pred(i,j,E,lam):
 c=m._fit(A[i],np.c_[y[i],E[i]]); r=y[i]-A[i]@c[:,0]; e=E-A@c[:,1:]; e/=np.maximum(1e-8,np.linalg.norm(e[i],axis=0)); return A[j]@c[:,0]+e[j]@np.linalg.solve(e[i].T@e[i]+lam*np.eye(e.shape[1]),e[i].T@r)
def scores(E,lam):return np.array([abs(y[p[52:]]-pred(p[:52],p[52:],E,lam)).mean() for p in splits])
result=[]
for k,b in [('base',np.empty((len(z),0))),*test.items()]:
 for lam in [.5,1.,2.,4.]:
  es=scores(np.c_[ebase,b],lam); result.append((.35*es[-1]+.35*max(es[1:-1])+.30*np.mean(es[1:-1]),k,lam,es[[0,-1]].tolist(),max(es[1:-1])))
print('best',*sorted(result)[:10],sep='\n'); print('base',*[r for r in result if r[1]=='base'],sep='\n')
