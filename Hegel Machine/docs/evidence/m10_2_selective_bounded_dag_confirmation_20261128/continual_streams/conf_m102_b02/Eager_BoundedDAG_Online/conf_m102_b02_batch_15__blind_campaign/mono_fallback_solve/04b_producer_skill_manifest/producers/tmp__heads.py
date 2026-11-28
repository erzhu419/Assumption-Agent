import numpy as np,itertools
from scipy.linalg import qr
from sklearn.linear_model import OrthogonalMatchingPursuit
p=np.load('/tmp/design.npz'); z,b,c,h=[p[k] for k in 'zbch']; Y=np.c_[p['y'],p['m']]; T=p['T']; f=p['f']; n=len(Y)
def features(coords):
 N=[]; F=[]
 def put(name,a):
  a=np.asarray(a); a=a[:,None] if a.ndim==1 else a
  for j,col in enumerate(a.T): N.append(name+str(j) if a.shape[1]>1 else name); F.append(col)
 for name,a in coords.items():
  put(name,a); put(name+'sq',a*a);put(name+'cu',a**3)
  for q in [.5,1,1.25,1.7,2.15,2.6,3.5,5]:
   for op,fn in [('s',np.sin),('c',np.cos),('t',np.tanh),('e',lambda v:np.exp(-v*v)),('r',lambda v:v/(1+v*v)),('a',np.arctan),('d',lambda v:np.sign(v)*np.maximum(abs(v)-.35,0)**1.5)]: put(f'{op}{q}:{name}',fn(q*a))
  for i,j in itertools.combinations(range(4),2):
   put(f'{name}{i}*{name}{j}',a[:,i]*a[:,j]);
   for q in [1,1.25,2.15]:
    for sign in [-1,1]:
     u=q*(a[:,i]+sign*a[:,j]); put(f's{q}({name}{i}{sign:+d}{name}{j})',np.sin(u));put(f't{q}({name}{i}{sign:+d}{name}{j})',np.tanh(u));put(f'c{q}({name}{i}{sign:+d}{name}{j})',np.cos(u));put(f'r{q}({name}{i}{sign:+d}{name}{j})',u/(1+u*u))
 for (k,a),(l,g) in itertools.combinations(coords.items(),2):
  for i,j in itertools.product(range(4),repeat=2):
   put(f'{k}{i}*{l}{j}',a[:,i]*g[:,j]);
 return N,np.array(F).T
if __name__=='__main__':
 for keys in ['z','zb','zc','zh','zbc','zbh','zch','zbch']:
  names,raw=features({k:p[k] for k in keys}); mu=raw[:52].mean(0); scale=np.maximum(raw[:52].std(0),1e-12); X=(raw-mu)/scale
  print(keys)
  for j in range(3):
   y=Y[:,j]; yc=y[:52]-y[:52].mean(); R=yc.copy(); active=[]; out=[]
   for k in range(12):
    idx=np.argmax(np.abs(X[:52].T@R));active.append(idx); v=np.linalg.lstsq(X[:52,active],yc,rcond=None)[0];R=yc-X[:52,active]@v
    pred=X[52:72,active]@v+y[:52].mean();out.append(round(np.mean(abs(pred-y[52:])),4))
   print(j,out,','.join(names[i] for i in active))
 print('y correlations with time',np.corrcoef(np.c_[Y,T].T)[0]);print('mode ranges',Y.min(0),Y.max(0))
