import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; L=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; O=lambda x:np.c_[np.ones(len(x)),x]; np.set_printoptions(precision=4,suppress=True)
a=s['panel_a']; g=np.max(abs(a['u']-a['v']),axis=1)<1; off,sc=np.stack([L(O(a['u'][g,k:k+1]),a['v'][g,k]) for k in range(4)]).T
op={}; rng=np.random.default_rng(45); folds=[(np.arange(56),np.arange(56,72))]+[(i[16:],i[:16]) for i in [rng.permutation(72) for _ in range(5)]]
for key,p,v in [('B','panel_b','v'),('C','panel_c','v1'),('D','panel_c','v2')]:
 u=s[p]['u']; z=s[p][v]; scores=[]
 for name,f in [('tanh',np.tanh),('atan',np.arctan),('sin',np.sin),('cos',lambda h:1-np.cos(h)),('rat',lambda h:h/(1+h*h)),('soft',lambda h:h/np.sqrt(1+h*h))]:
  F=lambda a,k:np.c_[O(a),f(a*k)]; r=least_squares(lambda k:(F(u,k)@L(F(u,k),z)-z).ravel(),[1.3],bounds=(.15,4)).x; c=L(F(u,r),z); scores.append((np.mean(abs(F(u,r)@c-z)),name,r,c,f));
 for err,n,r,c,_ in sorted(scores)[:3]:print(key,n,err,r,'linear to nonlinear row ratios',np.sum(c[1:5]*c[5:9],1)/np.sum(c[5:9]**2,1),'proportional residual',np.linalg.norm(c[1:5]-np.sum(c[1:5]*c[5:9],1)[:,None]/np.sum(c[5:9]**2,1)[:,None]*c[5:9]));
 err,n,r,c,f=min(scores,key=lambda h:h[0]); op[key]=lambda a,r=r,c=c,f=f:np.c_[O(a),f(a*r)]@c
 print(key,'coeff',c)
coords={'x':x,'inv':(x-off)/sc,'fwd':x*sc+off}; zz={**coords}
for n,u in coords.items():
 for k,f in op.items():
  zz[k+n]=f(u)
  for l,h in op.items():zz[l+k+n]=h(f(u))
def phi(u,m):
 if m==0:return O(u)
 if m==1:return np.c_[O(u),*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]]
 if m==2:return np.c_[phi(u,1),u*u]
 if m==3:return np.c_[O(u),u[:,0]*u[:,2]+u[:,1]*u[:,3]]
 if m==4:return np.c_[O(u),np.tanh(u)]
results=[]
for n,u in zz.items():
 for m in range(5):
  A=phi(u,m); errors=[]
  for i,j in folds:
   E=abs(A[j]@L(A[i],Y[i])-Y[j]); errors.append([E[:,0].mean(),E[:,1:].mean()]);
  results.append((n,m,np.array(errors)))
for k in range(2):
 print('target',k,[(n,m,round(e[0,k],5),round(max(e[:,k]),5)) for n,m,e in sorted(results,key=lambda h:np.mean(h[2][:,k]))[:12]])
fams={'exp':lambda k:-np.expm1(-k*t),'rat':lambda k:t/(1+k*t),'log':lambda k:np.log1p(k*t),'sqrt':lambda k:np.sqrt(1+k*t)-1,'t':lambda k:t,'t2':lambda k:t*t}
Z=Y[:,1:]; o=[]
for a,b in combinations(fams,2):
 def res(k):
  T=np.column_stack((fams[a](k[0]),fams[b](k[1]))); return (T@L(T,Z.T)-Z.T).ravel()
 r=min([least_squares(res,start,bounds=(.005,12)) for start in [[.5,1.5],[1.5,.5],[3,3]]],key=lambda q:q.cost); o.append((np.sqrt(np.mean(r.fun**2)),a,b,r.x))
print('optimized time pairs',sorted(o)[:9])
