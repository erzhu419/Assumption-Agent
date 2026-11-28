exec(open('/tmp/probe.py').read().split('rec=[]')[0])
from itertools import combinations_with_replacement
from sklearn.model_selection import KFold
rng=np.random.default_rng(1642)
splits += [(q[:52],q[52:]) for q in (rng.permutation(72) for _ in range(4))]
# Held-out arrays are scored only after each kernel is fitted.
base={'x':x,'z':z,'bz':bz,'bx':bx,'c1z':c1(z),'c1b':c1(bz),'c2z':c2(z),'c2b':c2(bz)}
rec=[]; NN=len(x); eye=np.eye(NN)
def keep(K, pars):
 E=[]; P=[]
 for i,o in splits:
  est=K[np.ix_(o,i)]@cho_solve(cho_factor(K[np.ix_(i,i)],check_finite=False),Y[i],check_finite=False); P.append(est); E.append(np.mean(abs(est-Y[o]),axis=0))
 rec.append((np.array(E),pars,P))
for name,X in base.items():
 if name in ['x','z','bz','bx']:
  params=[(m,h) for m in ['raw','tanh','atan','soft','sq','sin','sinh'] for h in ([0] if m=='raw' else [.35,.75,1.25,2.])]
 else: params=[('raw',0),('tanh',.5),('sq',1.)]
 for m,h in params:
  F={'raw':lambda:X,'tanh':lambda:np.tanh(h*X)/h,'atan':lambda:np.arctan(h*X)/h,'soft':lambda:X/(1+h*abs(X)),'sq':lambda:X/np.sqrt(1+h*X*X),'sin':lambda:np.sin(h*X)/h,'sinh':lambda:np.sinh(h*X)/h}[m]();
  D=(F[:,None,:]-F[None,:,:])**2; L=F@F.T; GX=(F[:,None,:]*F[None,:,:]); Q=L**2; D0=D.sum(2)
  for w in [.15,.3,.45,.65,.9,1.3,1.9]:
   A=np.exp(-w*D); add=A.mean(2); pair=(A.sum(2)**2-(A*A).sum(2))/12; triple=np.mean(np.stack([np.prod(A[:,:,[j for j in range(4) if j!=k]],axis=2) for k in range(4)]),axis=0); R=np.exp(-w*D0)
   for mode,R0 in enumerate([R, .5*R+.5*add, .3*R+.7*pair, .2*R+.8*triple, .1*R+.9*add, .3*R+.4*add+.3*pair, R*(1+.6*L),.6*R+.4*np.exp(-.22*w*D0)]):
    for q in [0.,.05,.2,.65]:
     G=1+L+q*Q+R0
     for g in [.0002,.0015,.006,.02]:keep(G+g*eye,(name,m,h,w,mode,q,g))
 print('done',name,flush=True)
E=np.array([r[0] for r in rec]); P=np.array([np.concatenate(r[2]) for r in rec]); Yval=np.concatenate([Y[o] for i,o in splits]); params=np.array([r[1] for r in rec]); np.savez('/tmp/final_grid13.npz',E=E,P=P,Y=Yval,params=params)
for j in range(2):
 for label,V in [('gate',E[:,0,j]),('oof',E[:,1:,j].mean(1)),('robust',.32*E[:,0,j]+.53*E[:,1:,j].mean(1)+.15*E[:,:,j].max(1))]:
  print('target',j,label)
  for k in np.argsort(V)[:12]:print(params[k].tolist(),np.round(E[k,:,j],5))
  for count in [10,30,100,300,700]:
   ii=np.argsort(V)[:count]; D=abs(P[ii,:,j].mean(0)-Yval[:,j]); idx=np.cumsum([len(o) for i,o in splits]); print('blend',count,[round(a.mean(),5) for a in np.split(D,idx[:-1])])
