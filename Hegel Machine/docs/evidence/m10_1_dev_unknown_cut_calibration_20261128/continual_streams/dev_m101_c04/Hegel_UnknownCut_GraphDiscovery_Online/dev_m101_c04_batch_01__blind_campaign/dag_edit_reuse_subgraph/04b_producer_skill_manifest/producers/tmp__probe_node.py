import numpy as np, pickle, runpy, itertools
from scipy.optimize import least_squares
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
z=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr']; y=s['train']['y']; r=s['train']['traj']; t=s['eval_times']; a,b,c,d=z.T
print('times',t,'z ranges',z.min(0),z.max(0),'trajectory svd',np.linalg.svd(r,compute_uv=False))
cols={'1':np.ones(len(z))}
for deg in (1,2,3):
 for inds in itertools.combinations_with_replacement(range(4),deg): cols[''.join('abcd'[i] for i in inds)]=np.prod(z[:,inds],axis=1)
for i,j in itertools.combinations(range(4),2):
 for k in (-2,-1,-.5,.5,1,2): cols[f'{i}+{k}*{j}']=z[:,i]+k*z[:,j]
for name,x in list(cols.items()):
 if name=='1':continue
 for f in (np.sin,np.cos,np.tanh,np.abs):cols[f'{f.__name__}({name})']=f(x)
 for k in (.2,.4,.5,1,2):
  for j in ('1','a','b','c','d'):
   if len(name)<=2:cols[f'{j}/(1+{k}*{name}^2)']=cols[j]/(1+k*x*x)
def fit(X,Y,itr,ite):
 w=np.linalg.lstsq(X[itr],Y[itr],rcond=None)[0]; return w,np.mean(np.abs(X[ite]@w-Y[ite]),axis=0)
tr=np.arange(52);te=np.arange(52,72)
for deg in (1,2,3):
 names=[n for n in cols if n=='1' or (len(n)<=deg and all(c in 'abcd' for c in n))];X=np.array([cols[n] for n in names]).T
 for label,Y in [('scalar',y),('traj-end',r[:,-1]),('traj',r)]:
  w,err=fit(X,Y,tr,te);print(deg,label,err,[(n,round(v,4)) for n,v in zip(names,w)] if Y.ndim==1 else '')
for form in ('exp','exp-shift','exp-offset'):
 def curve(p):
  if form=='exp':return -np.expm1(-p[0]*t)
  if form=='exp-shift':return -np.expm1(-p[0]*(t+p[1]))
  return p[1]-np.exp(-p[0]*t)
 def res(p):
  g=curve(p); amp=r[tr]@g/(g@g);return (amp[:,None]*g-r[tr]).ravel()
 v=least_squares(res,[.6] if form=='exp' else [.6,.1 if form=='exp-shift' else 1.]).x
 g=curve(v); amp=r@g/(g@g);print(form,v,g,'trajectory profile MAE',np.mean(np.abs(amp[:,None]*g-r),axis=0));print('amp linear',fit(np.c_[np.ones(len(z)),z],amp,tr,te))
X=np.column_stack(list(cols.values()));names=np.array(list(cols));
for Y in (y,r[:,-1]):
 selected=[0]; unused=np.arange(1,X.shape[1]); print('greedy')
 for _ in range(12):
  G=X[tr][:,selected];q=np.linalg.qr(G)[0]; E=X[tr][:,unused]; E=E-q@(q.T@E);res=Y[tr]-q@(q.T@Y[tr]); j=np.argmax(np.abs(E.T@res)/np.maximum(np.linalg.norm(E,axis=0),1e-9)); selected.append(unused[j]); unused=np.delete(unused,j); w,err=fit(X[:,selected],Y,tr,te); print(names[selected[-1]],round(float(err),5),np.round(w,3))
