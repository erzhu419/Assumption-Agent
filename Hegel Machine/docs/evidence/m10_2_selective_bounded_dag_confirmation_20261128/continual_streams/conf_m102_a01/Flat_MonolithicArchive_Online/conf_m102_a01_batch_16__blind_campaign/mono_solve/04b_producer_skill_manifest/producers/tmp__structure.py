exec(open('/tmp/grammar.py').read().split('_,_,V=')[0])
from scipy.optimize import least_squares
from scipy.stats import pearsonr
n=52
V=np.linalg.svd(t[:n],full_matrices=False)[2]; a=t@V[0]; b=t@V[1]
Q=np.column_stack([A(z),*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]])
for target in [y,a,b,t[:,-1]]:
 c=ls(Q[:n],target[:n]); r=target-Q@c; P=np.linalg.qr(Q[:n])[0]; loo=r[:n]/(1-(P*P).sum(1)); print('quad coef',c,'MAE train/loo/val',*[round(np.mean(abs(q)),5) for q in [r[:n],loo,r[n:]]]); print('resid',np.c_[x,r][np.argsort(abs(r))[-8:]])
for k,w in W.items():
 cols=[]; tags=[]
 def add(f,name):
  f=np.asarray(f).reshape(len(x),-1); cols.append(f); tags.extend([name+str(i) for i in range(f.shape[1])])
 ii,jj=np.triu_indices(4,1); d=np.c_[w,w[:,ii]+w[:,jj],w[:,ii]-w[:,jj]]
 add(w,'id'); add(w*w,'sq'); add(w[:,ii]*w[:,jj],'cross');
 for j in [.22, .6, 1.25, 2.5]:
  for name,f in [('sin',np.sin(j*d)),('tanh',np.tanh(j*d)),('cos',np.cos(j*d)),('soft',soft(d,j)),('rad',rad(d,j)),('rat',d/(1+j*d*d)),('rat2',1/(1+j*d*d)),('abs',abs(d))]:
   add(f,f'{name}{j}_'); add((w[:,:,None]*f[:,None,:]).reshape(len(w),-1),f'x{name}{j}_')
 for order in [3,4]:add(np.column_stack([np.prod(w[:,ids],axis=1) for ids in combinations_with_replacement(range(4),order)]),f'poly{order}_')
 F=np.column_stack(cols)
 for target in [y,a]:
  c=ls(Q[:n],target[:n]); r=target-Q@c; G=F-Q@ls(Q[:n],F[:n]); G/=np.maximum(np.linalg.norm(G[:n],axis=0),1e-10); scores=abs(G[:n].T@r[:n]); ix=np.argmax(scores); g=G[:,ix]; r1=r-g*(g[:n]@r[:n]); print(k,'y' if target is y else 'a',tags[ix],round(scores[ix]/np.linalg.norm(r[:n]),3),round(np.mean(abs(r1[n:])),4),end='; ')
 print()
# Profile a single nonlinear ridge atop the polynomial, with or without a multiplicative affine driver.
for goal in [y,a]:
 for k in ['x','z','h','zc0','zc1']:
  w=W[k]; w=w-w[:n].mean(0); Q0=Q if goal is y else A(x); P=np.linalg.qr(Q0[:n])[0]; r=goal[:n]-P@(P.T@goal[:n]); out=[]
  for name,fun in [('tanh',np.tanh),('sin',np.sin),('soft',lambda q:q/(1+abs(q))),('rad',lambda q:np.sqrt(1+q*q)-1),('rat',lambda q:q/(1+q*q)),('rat2',lambda q:1/(1+q*q)),('abs',abs)]:
   for inter in [False,True]:
    def feat(p):
     g=fun(A(w)@p)[:,None]; return g*A(w) if inter else g
    def loss(p):
     f=feat(p)[:n]; f=f-P@(P.T@f); return r-f@ls(f,r)
    starts=np.r_[np.random.default_rng(952).normal(size=(8,5)),np.c_[np.zeros(4),1.25*np.eye(4)]]
    best=min((least_squares(loss,p,bounds=(-6,6),max_nfev=60) for p in starts),key=lambda o:o.cost)
    f=np.c_[Q0,feat(best.x)]; pred=f@ls(f[:n],goal[:n]); out.append((np.sqrt(np.mean((pred[:n]-goal[:n])**2)),np.mean(abs(pred[n:]-goal[n:])),name,inter,best.x))
  print('profile','y' if goal is y else 'a',k,sorted(out,key=lambda q:q[0])[:4]);
