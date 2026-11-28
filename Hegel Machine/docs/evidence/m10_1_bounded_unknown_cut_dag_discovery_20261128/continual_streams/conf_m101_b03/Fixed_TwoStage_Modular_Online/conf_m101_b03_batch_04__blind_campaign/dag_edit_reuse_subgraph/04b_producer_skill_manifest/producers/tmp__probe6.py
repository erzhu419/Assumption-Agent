from p import *
from scipy.optimize import minimize_scalar
from itertools import product
U,S,V=np.linalg.svd(T,full_matrices=False); Z=np.c_[y,U[:,:2]*S[:2]]
B=s['panel_b']; C=s['panel_c'];
def aff(a):return np.c_[np.ones(len(a)),a]
def cy(a):return a*a[:,[1,2,3,0]]
def CB(a):return np.c_[np.ones(len(a)),cy(a)]
b=lambda a:aff(a)@np.linalg.lstsq(aff(B['u']),B['v'],rcond=None)[0]
f=lambda a:CB(a)@np.linalg.lstsq(CB(C['u']),C['v1'],rcond=None)[0]
u=C['u']; v=C['v2']; n2=len(u)
def d2(a):return np.c_[aff(a),a*a]
def d3(a):return np.c_[d2(a),a**3]
def d4(a):return np.c_[d3(a),a**4]
for name,dsg in [('d2',d2),('d3',d3),('d4',d4)]:
 A=dsg(u); Q=np.linalg.qr(A,mode='reduced')[0]; e=v-Q@(Q.T@v); print(name,np.std(e,0),np.linalg.lstsq(A,v,rcond=None)[0].round(4).T)
# Test separable quadratic deformation including saturation of the linear arguments.
for fun in (np.tanh,np.sin,np.arctan):
 def ff(a,k):return np.c_[np.ones(len(a)),fun(k*a)/k,a*a]
 def loss(k):
  A=ff(u,k); return np.mean((v-A@np.linalg.lstsq(A,v,rcond=None)[0])**2)
 r=minimize_scalar(loss,bounds=(.001,2),method='bounded');print(fun.__name__,r.x,np.sqrt(r.fun))
g=lambda a:d3(a)@np.linalg.lstsq(d3(u),v,rcond=None)[0]
folds=[np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]
def score(A):
 err=[]
 for j in folds:
  i=np.setdiff1d(np.arange(len(x)),j); c=np.linalg.lstsq(A[i],Z[i],rcond=1e-8)[0]; p=A[j]@c; err.append([np.mean(abs(p[:,0]-y[j])),np.mean(abs(p[:,1:]@V[:2]-T[j]))])
 return np.array(err)
base=np.c_[d2(x),cy(x)]
D={'x':x,'b':b(x),'f':f(x),'g':g(x),'gb':g(b(x)),'gf':g(f(x)),'fg':f(g(x)),'fb':f(b(x))}
candidates=[('base',base)]
for k,a in D.items():
 candidates.extend([(k,np.c_[base,a]),(k+' product',np.c_[base,a,(a[:,:,None]*x[:,None,:]).reshape(len(x),-1)]),(k+' squares',np.c_[base,a,a*a])])
for k1,k2 in [('f','g'),('f','b'),('f','gb'),('f','gf'),('f','fg'),('f','fb'),('f','x')]:
 a,c=D[k1],D[k2]; candidates.extend([(k1+k2,np.c_[base,a,c,a*c]),(k1+k2+' allprod',np.c_[base,a,c,(a[:,:,None]*c[:,None,:]).reshape(len(x),-1)])])
for k,A in candidates:
 E=score(A); print(k,np.linalg.matrix_rank(A), 'tail',E[0].round(5),'worst',E.max(0).round(5))
