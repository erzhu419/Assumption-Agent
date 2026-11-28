import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
pairs=list(itertools.combinations(range(4),2)); ij=np.array(pairs).T
def cross(u): return u[:,ij[0]]*u[:,ij[1]]
def one(u): return np.ones((len(u),1))
def lin(u): return np.c_[one(u),u]
def quadratic(u): return np.c_[lin(u),u*u,cross(u)]
def fit(A,Y): return np.linalg.lstsq(A,Y,rcond=None)[0]
def loo(A,Y):
 P=np.linalg.pinv(A); r=(Y-A@(P@Y))/(1-np.sum(A*P.T,axis=1)[:,None]);return np.sqrt(np.mean(r*r))
A=s['panel_a'];u,v=A['u'],A['v']; L=lin(u); w=np.ones(len(u))
for _ in range(12):
 c=fit(L*w[:,None],v*w[:,None]);r=np.linalg.norm(v-L@c,axis=1);w=1/np.sqrt(1+(r/(.5*np.median(r)))**2)
k=np.sort(np.argsort(r)[:-3]);u,v=u[k],v[k]
print('A quad coefficients (intercept,linear,squares,cross 01 02 03 12 13 23)\n',fit(quadratic(u),v))
for name,fn in [('quad',quadratic),('quad+cubes',lambda u:np.c_[quadratic(u),u**3]),('quad+sin',lambda u:np.c_[quadratic(u),np.sin(u)]),('quad+cos',lambda u:np.c_[quadratic(u),np.cos(u)]),('quad+sin+cos',lambda u:np.c_[quadratic(u),np.sin(u),np.cos(u)])]: print('A',name,loo(fn(u),v))
for pn,key in [('panel_b','v'),('panel_c','v2')]:
 p=s[pn]; a,b=p['u'],p[key]
 fam=[('tanh',lambda x,z:np.tanh(z*x)),('sin',lambda x,z:np.sin(z*x)),('cos',lambda x,z:np.cos(z*x)),('rat',lambda x,z:x/(1+z*x*x))]
 for name,fn in fam:
  for addname,add in [('none',one),('lin',lin)]:
   g=lambda z:loo(np.c_[add(a),fn(a,z)],b)
   res=minimize_scalar(g,bounds=(.1,3),method='bounded');print(pn,name,addname,round(res.x,5),round(res.fun,7))
Bc=lambda x:np.c_[one(x),np.tanh(x),np.tanh(.5*x)]
C1=lambda x:np.c_[one(x),cross(x)[:,[0,2,3,5]]]
C2=lambda x:np.c_[lin(x),x*x,np.cos(2*x)]
def pred(panel,key,basis,x):
 p=s[panel];return basis(x)@fit(basis(p['u']),p[key])
X=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];Y=np.c_[y,T];
AF=quadratic; co=fit(AF(u),v); aff=fit(lin(u),v)
z=(X-aff[0])@np.linalg.inv(aff[1:]); zi=z.copy()
for _ in range(12):zi+=(X-AF(zi)@co)@np.linalg.inv(aff[1:])
F={}
for name,x in [('raw',X),('aff',z),('inv',zi),('forward',AF(X)@co)]:
 B=pred('panel_b','v',Bc,x); c=pred('panel_c','v1',C1,x); d=pred('panel_c','v2',C2,x)
 forms={'x':lin(x),'q':quadratic(x),'b':B,'c':c,'d':d,'bc':B*c,'bd':B*d,'cd':c*d,'bs':B*B,'ds':d*d,'cs':c*c,'xb':x*B,'xc':x*c,'xd':x*d,'tanh':np.tanh(x),'sin':np.sin(x),'cub':x**3,'abs':np.abs(x),'rat':x/(1+x*x)}
 def stats(g):
  vals=[]
  for test in [np.arange(52,72),*np.array_split(np.random.default_rng(123).permutation(72),5)]:
   tr=np.setdiff1d(np.arange(len(x)),test);G=g[tr]; err=np.abs(Y[test]-g[test]@np.linalg.solve(G.T@G+.001*np.eye(G.shape[1]),G.T@Y[tr]));vals.append([err[:,0].mean(),err[:,1:].mean()])
  return np.round(vals[0],4),np.round(np.max(vals,axis=0),4)
 for base in ['x','q']:
  for ks in [[],['b','c','d'],['b','c','d','bc','bd','cd'],['b','c','d','bc','bd','cd','xb','xc','xd'],['tanh','cub','sin','rat'],['b','c','d','bc','bd','cd','tanh','cub'],*[[k] for k in ['b','c','d','bc','bd','cd','xb','xc','xd']]]:
   g=np.column_stack([forms[base]]+[forms[k] for k in ks]);print(name,base,'+'.join(ks) or '-',*stats(g))
