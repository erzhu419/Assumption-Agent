import pickle
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); x=np.vstack([tr['x'],s['test']['x']]); t=s['eval_times']
def D(*z): return np.column_stack((np.ones(len(z[0])),*z))
def fit(g,v): return np.linalg.lstsq(g,v,rcond=None)[0]
def mse(x): return np.mean(x*x)**.5
def trim(g,y):
 b=fit(g,y)
 for _ in range(12):
  idx=np.argsort(abs(g@b-y))[:int(.8*len(y))]; b=fit(g[idx],y[idx])
 return b,np.sort(abs(g@b-y))
p=s['panel_a']; u,v=p['u'],p['v']; maps=[]
for full in [False,True]:
 C=np.zeros((5,4)); errs=[]
 for j in range(4):
  ids=np.arange(5) if full else [0,j+1]; b,e=trim(D(u)[:,ids],v[:,j]); C[ids,j]=b; errs.append(e[[18,30,36,38,40,42,44]])
 print('A',full,C.round(4),np.array(errs).round(4)); maps.append((x-C[0])@np.linalg.inv(C[1:]))
def quad(z): return np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)])
def shrink(z,k): return np.sign(z)*np.maximum(abs(z)-k,0)**1.5
p=s['panel_b']; f=lambda k: mse(D(shrink(p['u'],k))@fit(D(shrink(p['u'],k)),p['v'])-p['v']); k=minimize_scalar(f,bounds=(.2,.5),method='bounded').x; print('B',k,f(k)); B=lambda z:D(shrink(z,k))@fit(D(shrink(p['u'],k)),p['v'])
pc=s['panel_c']; forms={'aff':(lambda z,k:D(z),(.01,2)), 'quad':(lambda z,k:D(z,quad(z)),(.01,2)), 'odd':(lambda z,k:D(-np.sign(z)*np.expm1(-k*abs(z))),(.01,5)), 'tanh':(lambda z,k:D(np.tanh(k*z)),(.01,5)), 'even':(lambda z,k:D(z,z*z/np.sqrt(1+k*z*z)),(.00001,5)), 'rat':(lambda z,k:D(z,z*z/(1+k*z*z)),(.00001,5)), 'oddrat':(lambda z,k:D(z/np.sqrt(1+k*z*z)),(.00001,5))}
ops=[]
for part in ['v1','v2']:
 opts=[]
 for key,(g,bnds) in forms.items():
  def loss(k):
   a=g(pc['u'],k); h=np.linalg.pinv(a); return mse((a@(h@pc[part])-pc[part])/(1-np.sum(a*h.T,axis=1))[:,None])
  q=minimize_scalar(loss,bounds=bnds,method='bounded').x; opts.append((loss(q),key,q)); print(part,key,round(q,5),round(loss(q),6))
 _,key,q=min(opts); g=forms[key][0]; c=fit(g(pc['u'],q),pc[part]); ops.append(lambda z,g=g,q=q,c=c:g(z,q)@c)
C,E=ops
def modes(k): return -np.expm1(-t[:,None]*np.asarray(k))
rr=least_squares(lambda k:(modes(k)@fit(modes(k),tr['traj'].T)-tr['traj'].T).ravel(),[.4,1.8]); print('rates',rr.x,'RMSE',mse(rr.fun)); T=np.column_stack((tr['y'],fit(modes(rr.x),tr['traj'].T).T))
def ridge(A,T,itr,ite,lam):
 g=A[itr]; c=g.mean(0); z=g-c; y=T[itr]; ym=y.mean(0); return (A[ite]-c)@z.T@np.linalg.solve(z@z.T+lam*np.eye(len(itr)),y-ym)+ym
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; rng=np.random.default_rng(124)
for j in range(3):
 ids=rng.permutation(n); splits.append((ids[:52],ids[52:]))
def scores(g,lam):
 o=[]
 for it,iv in splits:
  y=ridge(g,T,it,iv,lam); o.append([np.mean(abs(y[:,0]-T[iv,0])),np.mean(abs(y[:,1:]@modes(rr.x).T-tr['traj'][iv]))])
 return np.array(o)
if __name__=='__main__':
 res=[]
 for an,a in zip(['raw','diag','full'],[x,*maps]):
  b=B(a)
  for cn,z in [('a',a),('b',b)]:
   c,e=C(z),E(z)
   nodes={'a':a,'b':b,'c':c,'d':e}
   dot=[np.sum(nodes[i]*nodes[j],axis=1) for i,j in ['ac','ad','bc','bd','cd','ab']]
   for nd in ['ab','ac','ad','abc','abd','acd','bcd','abcd']:
    g0=np.column_stack([nodes[j] for j in nd]); cross=lambda i,j:(nodes[i][:,:,None]*nodes[j][:,None,:]).reshape(len(a),-1)
    for label,g in [('lin',g0),('dot',np.column_stack([g0,*dot])),('quad',np.column_stack([g0,quad(g0)])),('ac',np.column_stack([g0,cross('a','c')])),('ad',np.column_stack([g0,cross('a','d')])),('both',np.column_stack([g0,cross('a','c'),cross('a','d')]))]:
     for lam in [1e-6,1e-4,.002,.02,.15]:
      sc=scores(g,lam); res.append((sc,an,cn,nd,label,lam))
 for i in range(2):
  print('BEST',i)
  for sc,*tag in sorted(res,key=lambda a:np.max(a[0][:,i]))[:12]: print(tag,sc.round(4).T.tolist())
