from operators import *
from scipy.optimize import minimize_scalar

def probe(p,key):
 u,v=p['u'],p[key]; o=np.arange(len(u)); kw={}
 funcs={'sin':lambda u,k:np.sin(k*u),'tanh':lambda u,k:np.tanh(k*u),'rat':lambda u,k:u/(1+k*u*u)}
 extras={'none':lambda u:u[:,:0],'diag2':lambda u:u*u,'pairs':lambda u:np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)]),'trips':lambda u:np.column_stack([np.prod(u[:,j],axis=1) for j in it.combinations(range(4),3)])}
 for name,f in funcs.items():
  for ex,g in extras.items():
   def des(x,k):return np.c_[aff(x),f(x,k),g(x)]
   def learn(i):
    def objective(k):
     a=des(u[i],k); return np.mean((a@fit(a,v[i])-v[i])**2)
    k=minimize_scalar(objective,bounds=(.3,4),method='bounded').x; b=fit(des(u[i],k),v[i]);return k,b
   k,b=learn(o); e=v-des(u,k)@b; ee=[]
   for j in o:
    kj,bj=learn(o[o!=j]);ee.append(v[j]-des(u[j:j+1],kj)[0]@bj)
   print(key,name,ex,round(k,5),np.round([np.std(e),np.std(ee)],6))
   if name=='tanh' and ex=='none':print('coefficients:',np.round(b,4));print('resid sv:',np.linalg.svd(e,compute_uv=False))
   kw[name,ex]=(k,b,lambda x, k=k,b=b,f=f,g=g: np.c_[aff(x),f(x,k),g(x)]@b)
 return kw
B=probe(s['panel_b'],'v'); C=probe(s['panel_c'],'v1')
t=s['train']; x=coords(t['x']); Y=t['y']; T=t['traj']; ti=s['eval_times']; ids=np.arange(72); random=np.random.default_rng(801).permutation(ids); splits=[ids[52:],ids[56:],np.argsort(np.linalg.norm(x,axis=1))[-16:],*np.array_split(random,4)]
def score(A,V):
 out=[]
 for j in splits:
  i=np.setdiff1d(ids,j); out.append(np.mean(np.abs(A[j]@fit(A[i],V[i])-V[j])))
 return np.round([out[0],out[1],out[2],max(out[3:])],5)
for name,extra in [('sin','none'),('tanh','none'),('rat','none'),('sin','pairs'),('tanh','pairs'),('tanh','diag2')]:
 z=B[name,extra][2](x); a=pol(z,2); b=fit(a[:52],Y[:52]); print('downstream',name,extra,score(a,Y),score(a,T),score(aff(z),T));print('label poly coeff',np.round(b,4)); print('traj coeff quadratic final',np.round(fit(a[:52],T[:52,-1]),4));
 def basis(p):return -np.expm1(-np.exp(p)[:,None]*ti)
 def res(p):
  q=basis(p);return (T[:52]-(T[:52]@np.linalg.pinv(q))@q).ravel()
 r=least_squares(res,np.log([.35,1.8]),bounds=(-5,3)).x;q=basis(r);am=T@np.linalg.pinv(q); print('relax rates',np.exp(r),'resid',np.std(T-am@q),'amplitudes',np.round(fit(a[:52],am[:52]),4).T)
