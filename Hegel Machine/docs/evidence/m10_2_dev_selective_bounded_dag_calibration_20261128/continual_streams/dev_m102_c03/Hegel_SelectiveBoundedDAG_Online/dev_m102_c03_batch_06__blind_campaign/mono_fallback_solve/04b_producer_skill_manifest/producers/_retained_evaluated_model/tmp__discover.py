import numpy as np,pickle
from itertools import combinations
from scipy.optimize import least_squares
from ops import _branches,_one,_fit,_calibration,_inverse
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; y=tr['y']; n=52; x=tr['x']; z=_inverse(x,_calibration(s['panel_a'])); c,d=_branches(s,x); p=s['panel_b']; b=_one(z)@_fit(_one(p['u']),p['v']); fit=_fit
ma=lambda a:np.mean(abs(a),axis=0)
rm=lambda a:np.sqrt(np.mean(a*a))
for label,fun,p0 in [('exp',lambda a:-np.expm1(-t*a),[.4,1.8,4.8]),('sin',lambda a:np.sin(t*a),[.4,1.8,1.]),('cos',lambda a:1-np.cos(t*a),[.4,1.8,1.]),('damped',lambda a:np.exp(-t*a[1])*np.sin(t*a[0]),[.4,1.8,2,1]),('linear',lambda a:t,[.4,1.8]),('t_exp',lambda a:t*np.exp(-t*a),[.4,1.8,2.])]:
 def qf(a):return np.column_stack([-np.expm1(-t[:,None]*a[:2]),fun(a[2:] if len(a)==4 else (a[2] if len(a)==3 else 0))])
 def err(a):
  q=qf(a);return (q@fit(q,T[:n].T)-T[:n].T).ravel()
 r=least_squares(err,p0,max_nfev=200).x; q=qf(r); print('time',label,r,rm(err(r)),rm(q@fit(q,T[n:].T)-T[n:].T));
 if label=='exp':A=fit(q,T.T).T; Q=q
Y=np.column_stack([y,A]); bas=np.column_stack([np.ones(len(x)),c,d,(c*d).sum(1)]); F={}; blocks={}
def block(name,f):
 f=np.reshape(f,(len(x),-1)); blocks[name]=f
 for j,g in enumerate(f.T):F[f'{name}:{j}']=g
G={'z':z,'b':b,'c':c,'d':d}
for k,g in G.items():
 for name,fun in [('id',lambda u:u),('sq',np.square),('cub',lambda u:u**3),('tanh',np.tanh),('sat',lambda u:u/(1+abs(u))),('g',lambda u:np.exp(-u*u)),('ab',abs),('rq',lambda u:u/(1+u*u))]:block(k+name,fun(g))
 for w in (.75,1.25,1.8,2.4,3.2,4.8):
  block(k+'sin'+str(w),np.sin(w*g));block(k+'cos'+str(w),np.cos(w*g))
 for op,sg in [('p',1),('m',-1)]:
  a=np.column_stack([g[:,i]+sg*g[:,j] for i,j in combinations(range(4),2)]);
  for nm,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('abs',abs),('sat',lambda u:u/(1+abs(u)))]:block(k+op+nm,f(a))
 for power in (2,3,4):block(k+'prod'+str(power),np.column_stack([np.prod(g[:,j],axis=1) for j in combinations(range(4),power)]))
for a,bn in combinations(G,2):
 u,v=G[a],G[bn]; block(a+bn+'outer',u[:,:,None]*v[:,None,:]);
 for k in (.25,.5,1,2):
  block(a+bn+'div'+str(k),u[:,:,None]/(1+k*v[:,None,:]**2)); block(bn+a+'div'+str(k),v[:,:,None]/(1+k*u[:,None,:]**2))
 for name,w in [('plus',u+v),('minus',u-v),('prod',u*v),('blend',u*np.tanh(v))]:
  for nm,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('abs',abs),('sat',lambda a:a/(1+abs(a)))]:block(a+bn+name+nm,f(w))
names=np.array(list(F)); D=np.column_stack(list(F.values())); print('dictionary',D.shape)
for yo,tar in enumerate(Y.T):
 print('TARGET',yo)
 for init,base in [('intercept',bas[:,:1]),('base',bas),('base+observed-mode',np.c_[bas,np.delete(Y,yo,axis=1)]),('base+z',np.c_[bas,z])]:
  errs=[]; found=[]; H=base.copy()
  for i in range(19):
   B=fit(H[:n],tar[:n]); R=tar-H@B; errs.append((ma(R[:n]),ma(R[n:]))); d1=D[:n]-H[:n]@fit(H[:n],D[:n]); den=np.linalg.norm(d1,axis=0); score=abs(R[:n]@d1)/np.maximum(den,1e-10); score[den<1e-7]=0; j=np.argmax(score); found.append(names[j]); H=np.column_stack([H,D[:,j]])
  print(init,'steps',[(i,found[i-1] if i else '-',*(round(x,4) for x in errs[i])) for i in range(10)],'best',np.min(np.array(errs)[:,1]))
 print('blocks',sorted([(ma(tar[n:]-(h:=np.c_[bas,f])[n:]@fit(h[:n],tar[:n])),k,ma(tar[:n]-h[:n]@fit(h[:n],tar[:n]))) for k,f in blocks.items()])[:12])
