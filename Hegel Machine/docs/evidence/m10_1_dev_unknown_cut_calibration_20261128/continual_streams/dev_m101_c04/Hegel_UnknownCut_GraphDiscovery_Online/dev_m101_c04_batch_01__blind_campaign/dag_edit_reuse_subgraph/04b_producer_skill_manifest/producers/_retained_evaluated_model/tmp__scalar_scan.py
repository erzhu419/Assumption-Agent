import pickle,runpy,itertools,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];z=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr'];y=s['train']['y'];a,b,c,d=z.T
L=np.c_[np.ones(len(z)),z]; Q=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)])
tr=np.arange(52);te=np.arange(52,72)
def score(A):
 w=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0];e=A@w-y;return np.sqrt(np.mean(e[tr]**2)),np.mean(abs(e[te])),w
print('linear',score(L)[:2]);print('structured interactions',score(np.c_[L,a*c,b*d])[:2]);print('quadratic',score(np.c_[L,Q])[:2]);
F={};
for x,v in zip('abcd',z.T): F[x]=v
for i,j in itertools.combinations(range(4),2):
 for k in (-2,-1,-.5,.5,1,2):F[f'{i}:{k}:{j}']=z[:,i]+k*z[:,j]
for i,j in itertools.combinations_with_replacement(range(4),2): F['abcd'[i]+'abcd'[j]]=z[:,i]*z[:,j]
for i,j,k in itertools.combinations_with_replacement(range(4),3):F['abcd'[i]+'abcd'[j]+'abcd'[k]]=z[:,i]*z[:,j]*z[:,k]
F['ac+bd']=a*c+b*d;F['ac-bd']=a*c-b*d
S={};
for n,v in F.items():
 S[n]=v
 for q in (.5,1,1.5,2,3):
  for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('softsign',lambda x:x/(1+abs(x)))]:S[f'{fn}({q}*{n})']=f(q*v)
  if len(n)<3 or ':' in n:
   for nn,vv in [('1',1),*zip('abcd',z.T),('ac',a*c),('bd',b*d),('ac+bd',a*c+b*d),('ac-bd',a*c-b*d)]:S[f'{nn}/(1+{q}*{n}^2)']=vv/(1+q*v*v)
X=np.column_stack(list(S.values()));names=np.array(list(S));
for base in [L,np.c_[L,a*c,b*d],np.c_[L,Q]]:
 A=base.copy(); selected=[]
 for step in range(5):
  U=np.linalg.svd(A[tr],full_matrices=False)[0][:,:np.linalg.matrix_rank(A[tr])];V=X[tr]-U@(U.T@X[tr]);norm=np.linalg.norm(V,axis=0);gain=np.abs(V.T@(y[tr]-U@(U.T@y[tr])))/np.maximum(norm,1e-8);gain[norm<1e-7]=0
  ids=np.argsort(-gain)[:4];print('base',base.shape[1],'step',step,[(names[j],*np.round(score(np.c_[A,X[:,j]])[:2],5)) for j in ids]);j=ids[0];A=np.c_[A,X[:,j]];selected.append(names[j])
# Fit one compact nonlinear ridge component using variable projection.
for fname,f in [('sin',np.sin),('tanh',np.tanh),('softsign',lambda v:v/(1+abs(v))),('rational',lambda v:1/(1+v*v))]:
 hits=[];rng=np.random.default_rng(1)
 for base in [L,np.c_[L,a*c,b*d]]:
  def mat(p):return np.c_[base,f(L@p)]
  def res(p):
   A=mat(p)[tr];return A@np.linalg.lstsq(A,y[tr],rcond=None)[0]-y[tr]
  for p in rng.normal(0,.8,(24,5)):
   opt=least_squares(res,p,bounds=(-7,7),max_nfev=160);rmse,err,w=score(mat(opt.x));hits.append((rmse,err,len(base.T),np.round(opt.x,3),np.round(w,3)))
 print(fname,sorted(hits,key=lambda h:h[0])[:1])
