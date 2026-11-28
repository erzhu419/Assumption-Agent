from operators import *
import importlib.util
from scipy.optimize import least_squares
m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'));sp.loader.exec_module(m)
x=s['train']['x'];z=s['train']['traj'];y=s['train']['y']; c,b,v,w=m.coordinates(s,x)
p=s['panel_c'];u=p['u'];vfun=lambda q: basis(np.arctan(2.12*q),1)@fit(basis(np.arctan(2.12*u),1),p['v1']); bfun=lambda q:basis(q,1)@fit(basis(s['panel_b']['u'],1),s['panel_b']['v']); wbase=lambda q:np.c_[basis(q,1),q*q/(1+.1616*q*q)];wfun=lambda q:wbase(q)@fit(wbase(u),p['v2']);
qs={'x':x,'c':c,'b':b,'bc':bfun(c),'v':v,'w':w,'wb':bfun(wfun(c)),'cbv':vfun(bfun(c)),'cbw':wfun(bfun(c))}
holds=[np.arange(52,72), np.arange(20),np.argsort(np.linalg.norm(x,axis=1))[-20:],np.argsort(x[:,0])[:20],np.argsort(x[:,3])[-20:]]
Y=[y,z@np.linalg.svd(z,full_matrices=False)[2][0]]
funcs={'raw':lambda q,a:q,'rsq':lambda q,a:q/np.sqrt(1+a*q*q),'rat':lambda q,a:q/(1+a*q*q),'atan':lambda q,a:np.arctan(a*q)/a,'tanh':lambda q,a:np.tanh(a*q)/a,'sin':lambda q,a:np.sin(a*q)/a}
for name,A in qs.items():qs[name]=A
for j,T in enumerate(Y):
 R=[]
 tr=np.arange(52)
 for name,A in qs.items():
  for fn,f in funcs.items():
   for alpha in ([0.] if fn=='raw' else [.16,.22,.5,1.,1.25,2.12]):
    q=f(A,alpha)
    for d in (2,3):
     D=basis(q,d)
     for den in [np.ones(len(x)),1+.22*(q*q).sum(1),1+.22*(A*A).sum(1),1+.22*(q.sum(1))**2]:
      P=D/den[:,None]
      for ridge in (.0001,.02):
       P0=P[tr];beta=np.linalg.solve(P0.T@P0+ridge*np.eye(P.shape[1]),P0.T@T[tr]); e=np.mean(abs(P@beta-T)[52:]); train=np.mean(abs(P@beta-T)[tr]); R.append((e,train,name,fn,alpha,d,den,ridge,P))
 print('target',j)
 for criteria in [lambda a:a[0],lambda a:a[1]]:
  for e,train,name,fn,alpha,d,den,ridge,P in sorted(R,key=criteria)[:8]:
   E=[]
   for h in holds:
    k=np.setdiff1d(np.arange(len(x)),h);P0=P[k];beta=np.linalg.solve(P0.T@P0+ridge*np.eye(P.shape[1]),P0.T@T[k]);E.append(np.mean(abs(P[h]@beta-T[h])))
   print(name,fn,alpha,d,ridge,'den range',round(den.max(),3),'train',round(train,5),'hold',np.round(E,5))
