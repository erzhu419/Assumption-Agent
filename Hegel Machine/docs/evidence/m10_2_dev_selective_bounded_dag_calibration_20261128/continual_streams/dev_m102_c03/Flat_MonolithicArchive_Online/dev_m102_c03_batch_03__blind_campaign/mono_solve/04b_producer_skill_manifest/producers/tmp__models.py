import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; n=52
np.set_printoptions(precision=5,suppress=True)
def aff(x): return np.c_[np.ones(len(x)),x]
def af(x): return np.c_[aff(x),np.roll(x,-1,axis=1)*x,x**3]
u=s['panel_a']['u']; v=s['panel_a']['v']; h=af(u); k=np.linalg.norm(v-aff(u)@np.linalg.lstsq(aff(u),v,rcond=None)[0],axis=1)<3
b=np.linalg.lstsq(h[k],v[k],rcond=None)[0]; print('A matrix',b,'sd',np.std(h[k]@b-v[k],axis=0))
z=np.array([least_squares(lambda w:(af(w[None,:])@b).ravel()-a,a).x for a in x]); print('inverse max',np.max(abs(af(z)@b-x)), np.min(z,axis=0),np.max(z,axis=0))
p=s['panel_b']; B=aff(z)@np.linalg.lstsq(aff(p['u']),p['v'],rcond=None)[0]
def panel(k,fun,init,bounds):
 u=s['panel_c']['u']; v=s['panel_c'][k];
 def project(p,out=False):
  F=fun(u,p); b=np.linalg.lstsq(F,v,rcond=1e-10)[0]; return fun(z,p)@b if out else (F@b-v).ravel()
 o=least_squares(project,np.array(init)*np.ones(4),bounds=bounds,max_nfev=100); return np.sqrt(np.mean(o.fun**2)),o.x,project(o.x,1)
funcs={'soft':(lambda a,p:a/(1+abs(a*p)),.8,(.02,5)), 'tanh':(lambda a,p:np.tanh(a*p),1.,(.1,6)), 'sin':(lambda a,p:np.sin(a*p),2.,(.1,6)), 'cos':(lambda a,p:np.cos(a*p),2.,(.1,6)), 'exp':(lambda a,p:np.exp(a*p),1.,(.02,4)), 'oddquot':(lambda a,p:a/(1+(a*p)**2),1.,(.02,5)), 'gauss':(lambda a,p:np.exp(-(a*p)**2),1.,(.02,5))}
cs=[]
for key in ['v1','v2']:
 out=[]
 for name,(g,i,lim) in funcs.items():
  for use_a in [0,1]:
   f=lambda a,p:np.c_[aff(a),g(a,p)] if use_a else aff(g(a,p))
   e,r,c=panel(key,f,i,lim); out.append((e,name,use_a,r,c))
 out.sort(key=lambda q:q[0]); print(key,[(e,name,use_a,r) for e,name,use_a,r,c in out[:6]]); cs.append(out[0][-1])
C,D=cs
q=-np.expm1(-t[:,None]*[.4029,1.8123]); a=np.linalg.lstsq(q,T.T,rcond=None)[0].T
Y=np.c_[y,a]
def F(a):return np.c_[aff(a),*[a[:,i]*a[:,j] for i in range(4) for j in range(i,4)]]
def score(name,f):
 b=np.linalg.lstsq(f[:n],Y[:n],rcond=1e-8)[0]; e=f@b-Y; print(name,f.shape[1],np.mean(abs(e[:n]),0),np.mean(abs(e[n:]),0),np.mean(abs(e[n:,1:]@q.T)))
for name,w in [('x',x),('z',z),('B',B),('C',C),('D',D)]:
 score(name+' linear',aff(w));score(name+' quadratic',F(w));score(name+' with tanh',np.c_[F(w),F(np.tanh(w))[:,1:]])
for i,j in combinations(range(4),2):
 w=[z,B,C,D]; f,g=w[i],w[j]; score(f'{i} {j} mix',np.c_[aff(z),f,g,np.einsum('ni,nj->nij',f,g).reshape(-1,16)]);score(f'{i} {j} quad',np.c_[F(f),F(g)[:,1:]])
