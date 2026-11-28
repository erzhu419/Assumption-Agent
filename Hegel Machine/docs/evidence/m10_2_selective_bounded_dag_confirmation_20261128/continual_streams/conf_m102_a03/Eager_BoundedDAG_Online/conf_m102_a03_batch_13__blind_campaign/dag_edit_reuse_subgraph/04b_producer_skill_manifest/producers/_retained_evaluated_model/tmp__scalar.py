import pickle,numpy as np,sys,itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; A=f._affine(u); x=f._inverse(s['train']['x'],f._fit_sensor(s['panel_a'])); y=s['train']['y']; tr=s['train']['traj']; U,D,V=np.linalg.svd(tr,full_matrices=False); H=np.c_[y,U[:,:2]*D[:2]]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for n in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),n)])
def cv(A,H,ids=None):
 if ids is None:ids=np.arange(len(A))
 i=ids[:-20];j=ids[-20:];return np.round(abs(A[j]@fit(A[i],H[i])-H[j]).mean(0),5)
def aff(x):return np.c_[np.ones(len(x)),x]
def bbase(q):return np.c_[aff(q),np.cos(2.6*q)]
def cbase(q):return np.c_[aff(q),np.tanh(2.15*q)]
def dbase(q):return np.c_[aff(q),np.cos(1.25*q)]
p=s['panel_b'];pc=s['panel_c'];z=bbase(x)@fit(bbase(p['u']),p['v']); c=cbase(x)@fit(cbase(pc['u']),pc['v1']); d=dbase(x)@fit(dbase(pc['u']),pc['v2']);
from scipy.optimize import minimize_scalar
P=poly(z,3); Q=poly(z,2); L=aff(z)
print('Cubic coefficients',list(zip([j for n in range(4) for j in itertools.combinations_with_replacement(range(4),n)],np.round(fit(P,y),4))))
r=y-L@fit(L,y)
print('z ranges:',z.min(0),z.max(0)); print('r stats',r.min(),r.max(),np.mean(abs(r)))
scores=[]
terms={'bilin':z[:,0]*z[:,2]+z[:,1]*z[:,3], 'diff':z[:,0]*z[:,2]-z[:,1]*z[:,3], 'linreadout':L@fit(L,y), 'norm':np.linalg.norm(z,axis=1)}
terms.update({str(i):z[:,i] for i in range(4)})
terms.update({f'{i}{j}':z[:,i]*z[:,j] for i,j in itertools.combinations(range(4),2)})
terms.update({f'{i}+{j}':z[:,i]+z[:,j] for i,j in itertools.combinations(range(4),2)})
terms.update({f'{i}-{j}':z[:,i]-z[:,j] for i,j in itertools.combinations(range(4),2)})
for num, q in list(terms.items()) + [('quad',None)]:
 for den, w in terms.items():
  for name,fun,bnd in [('rational',lambda k:1/(1+k*w*w),(.00001,20)),('linear',lambda k:1+k*w,(-2,2)),('cos',lambda k:np.cos(k*w),(.01,5)),('sin',lambda k:np.sin(k*w),(.01,5))]:
   def basis(k):
    m=fun(k); return np.c_[L, (Q[:,5:]*m[:,None] if q is None else q*m)]
   def err(k):
    B=basis(k); return np.mean((y-B@fit(B,y))**2)
   op=minimize_scalar(err,bounds=bnd,method='bounded');scores.append((op.fun**.5,num,den,name,op.x,cv(basis(op.x),y)))
print('Top scalar grammars'); print(*sorted(scores)[:22],sep='\n')
