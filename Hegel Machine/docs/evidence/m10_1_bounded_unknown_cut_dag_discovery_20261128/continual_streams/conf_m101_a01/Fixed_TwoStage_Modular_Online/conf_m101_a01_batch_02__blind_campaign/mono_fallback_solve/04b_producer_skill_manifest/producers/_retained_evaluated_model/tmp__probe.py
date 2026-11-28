import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];T=s['train'];x=T['x'];y=T['y'];tr=T['traj'];t=s['eval_times']; np.set_printoptions(precision=5,suppress=True)
def af(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,c],axis=1) for i in range(1,d+1) for c in combinations_with_replacement(range(x.shape[1]),i)])
def reg(u,v):return np.linalg.lstsq(u,v,rcond=None)[0]
a=s['panel_a'];u=af(a['u']);v=a['v'];good=np.ones(len(u),bool)
for i in range(10):
 w=reg(u[good],v[good]);r=np.linalg.norm(u@w-v,axis=1);good=r < max(3*np.median(r),.001)
print('A',w,'r sorted',np.sort(r)); print('A svd',np.linalg.svd(w[1:])[1]);
b=s['panel_b'];B=reg(af(b['u']),b['v']);print('B svd',np.linalg.svd(B[1:])[1]); c=s['panel_c'];
for key in ['v1','v2']:
 u=c['u'];v=c[key];print('C',key,'range',v.min(0),v.max(0))
 for fun in [lambda v:v, np.arctanh,np.arcsin]:
  if np.max(abs(v))>=1 and fun!=np.arctanh and fun!=np.arcsin:pass
  elif np.max(abs(v))>=1:continue
  f=fun(v);print(fun,np.sqrt(np.mean((af(u)@reg(af(u),f)-f)**2)))
 for d in [1,2,3]:
  rr=[np.mean(abs(poly(u,d)[i]@reg(np.delete(poly(u,d),i,0),np.delete(v,i,0))-v[i])) for i in range(len(u))];print(d,np.mean(rr))
U,S,V=np.linalg.svd(tr,full_matrices=False);print('raw traj svd',S,'basis',V[:3]);
funs={'rise+sin':lambda p:np.stack([-np.expm1(-p[0]*t),np.exp(-p[1]*t)*np.sin(p[2]*t)]),'rise+dosc':lambda p:np.stack([-np.expm1(-p[0]*t),np.exp(-p[1]*t)*(1-np.cos(p[2]*t))]),'sin+dosc':lambda p:np.stack([np.exp(-p[0]*t)*np.sin(p[2]*t),np.exp(-p[1]*t)*(1-np.cos(p[2]*t))]),'rise+rise':lambda p:np.stack([-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)]),'rise+tsin':lambda p:np.stack([-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)*np.sin(p[2]*t)])}
for k,f in funs.items():
 def err(p):
  g=f(p);return (tr-reg(g.T,tr.T).T@g).ravel()
 opt=least_squares(err,[1.7,.2,1.7],bounds=([.0001,0,.0001],[10,5,5]));print(k,opt.x,np.std(opt.fun),np.std(err([1.7,.2,1.7])))
for name,z in [('raw',x),('A',af(x)@w),('Ai',(x-w[0])@np.linalg.inv(w[1:])),('AB',af(af(x)@w)@B),('AiB',af((x-w[0])@np.linalg.inv(w[1:]))@B),('B',af(x)@B)]:
 print(name,'range',z.std(0));
 for d in (1,2,3):
  F=poly(z,d); v=np.c_[y,U[:,:2]*S[:2]];e=F[52:]@reg(F[:52],v[:52])-v[52:];print(d,np.mean(abs(e),0))
