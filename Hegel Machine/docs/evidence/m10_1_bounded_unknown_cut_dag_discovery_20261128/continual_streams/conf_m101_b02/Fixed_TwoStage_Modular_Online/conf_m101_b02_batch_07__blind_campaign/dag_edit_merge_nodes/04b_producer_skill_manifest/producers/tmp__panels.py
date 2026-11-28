import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']

def terms(x,ts): return np.column_stack([np.prod(x[:,t],axis=1) for t in ts])
def fit(d,y): return np.linalg.lstsq(d,y,rcond=None)[0]
def score(d,y):
 u,z,_=np.linalg.svd(d,full_matrices=False);u=u[:,z>z[0]*1e-10];r=y-u@(u.T@y);return np.mean(abs(r/np.maximum(1-np.sum(u*u,axis=1)[:,None],.00001))),np.sqrt(np.mean(r*r))
base=[()]+[(j,) for j in range(4)];pairs=list(it.combinations(range(4),2)); sq=[(j,j) for j in range(4)];cu=list(it.combinations_with_replacement(range(4),3));tri=list(it.combinations(range(4),3)); pure=[(j,j,j) for j in range(4)]; cx=[(0,1),(0,3),(1,2),(2,3)]
p=s['panel_a'];d=terms(p['u'],base);a=np.zeros((5,4));a[1:]=np.eye(4)
for _ in range(12):
 w=1/np.sqrt(1+np.sum((d@a-p['v'])**2,axis=1)/.02**2);a=fit(d*w[:,None],p['v']*w[:,None])
print('A',a.round(5),'residual sorted',np.sort(np.linalg.norm(d@a-p['v'],axis=1))[-5:])
p=s['panel_c'];x=p['u'];d=terms(x,base+sq);print('C1 compact',score(terms(x,[()]+cx),p['v1']),'C2 squares',score(d,p['v2']));print('C2 compact coefs',fit(d,p['v2']).round(4));print('C2 residuals',score(np.column_stack((d,x**3)),p['v2']))
p=s['panel_b'];x=p['u'];y=p['v'];print('B empirical affine',fit(terms(x,base),y).round(3));results=[]
for tag,t in [('lin',[]),('pair',pairs),('cx',cx),('sq',sq),('tri',tri),('pure',pure),('pure+tri',pure+tri),('pair+pure',pairs+pure),('sq+pure',sq+pure)]:
 d=terms(x,base+t);results.append((score(d,y),tag));
 for f in np.arange(.5,6.01,.1):
  for nm,h in [('sin',np.sin(f*x)),('cos',np.cos(f*x)),('tanh',np.tanh(f*x)),('rat',x/(1+f*x*x)),('sqrt',x/np.sqrt(1+f*x*x))]: results.append((score(np.column_stack((d,h)),y),(tag,nm,round(f,2))))
print('B top library',sorted(results)[:10]);print('B base terms',sorted(z for z in results if isinstance(z[1],str)))
# Find a common sparse residual basis with column-normalized forward selection.
sums=np.column_stack([x[:,i]+k*x[:,j] for i,j in pairs for k in (1,-1,.5,-.5,2,-2)]); lib={str(t):terms(x,[t])[:,0] for t in pairs+sq+cu}
for name,z in [('x',x),('mix',sums)]:
 for f in (.5,1,1.2,1.5,2,2.5,3,3.5,4):
  for nm,h in [('sin',np.sin(f*z)),('cos',np.cos(f*z)),('tanh',np.tanh(f*z)),('rat',z/(1+f*z*z))]:
   for j in range(z.shape[1]):lib[str((name,nm,f,j))]=h[:,j]
d0=terms(x,base); names=list(lib); D=np.column_stack(list(lib.values()));
for j in range(4):
 d=d0; picks=[]
 for _ in range(5):
  q=np.linalg.qr(d)[0]; r=y[:,j]-q@(q.T@y[:,j]); Z=D-q@(q.T@D); norm=np.linalg.norm(Z,axis=0); i=np.argmax(np.where(norm>1e-8,abs(Z.T@r)/np.maximum(norm,1e-8),0)); d=np.column_stack((d,D[:,i]));picks.append((names[i],tuple(round(v,5) for v in score(d,y[:,[j]]))))
 print('B sparse',j,picks)
