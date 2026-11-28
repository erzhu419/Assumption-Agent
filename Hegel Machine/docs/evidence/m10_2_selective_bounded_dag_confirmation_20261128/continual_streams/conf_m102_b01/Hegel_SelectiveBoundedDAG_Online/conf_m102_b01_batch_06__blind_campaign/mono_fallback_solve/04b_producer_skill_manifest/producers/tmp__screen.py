from graph import *
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
s0=s; n=52; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; print('times',s['eval_times']); print('mean',Y.mean(0),'std',Y.std(0),'corr y',np.corrcoef(Y.T)[0]);
W=graphs(s,tr['x']); z,A,B,C,D=ops(s,tr['x']); b=B(z);c=C(z);d=D(z);e=C(b); f=D(b); g=B(c)
for k,v in {'base':z,'b':b,'c':c,'d':d,'e':e,'f':f,'g':g}.items():
 print(k,'corr',np.round(np.corrcoef(np.c_[v,Y[:,[0,1,-1]]].T)[:4,4:],3))
W.update({'eb':np.c_[e,b], 'ef':np.c_[e,f], 'ebf':np.c_[e,b,f], 'ebz':np.c_[e,b,z], 'Bf':np.c_[b,f], 'eD':np.c_[e,d]})
for u in [z,b,c,d,e,f,g]:
 pass
rows=[]
def score(p): return np.mean(abs(p-Y),0)[[0,*range(1,9)]].reshape(-1)
for k,w in W.items():
 w=w/np.maximum(.05,np.sqrt(np.mean(w[:n]**2,0)))
 for j,fn in enumerate([lambda v:v,np.tanh,np.arctan,np.sin,lambda v:v/(1+.22*v*v)]):
  v=fn(w); Vs=[('p2',poly(v,2)),('p3',poly(v,3))] if w.shape[1]<=8 else [('p2',poly(v,2))]
  Vs += [('p2+B',np.c_[poly(v,2),b]), ('p2+both',np.c_[poly(v,2),b,e])]
  for q,F in Vs:
   U,S,V=np.linalg.svd(F[:n],full_matrices=False); G=F@V.T; R=U.T@Y[:n]
   for reg in [.0001,.003,.03,.3,2.]:
    wt=S/(S*S+reg); P=G@(wt[:,None]*R); lev=(U**2)@(S*wt); loo=(Y[:n]-P[:n])/(1-lev[:,None]); er=np.mean(abs(P[n:]-Y[n:]),0); cv=np.mean(abs(loo),0); rows.append((cv[0]+cv[1:].mean(),er[0],er[1:].mean(),cv[0],cv[1:].mean(),k,j,q,reg))
for j in [0,1,2]:print('best',j,*sorted(rows,key=lambda a:a[j])[:9],sep='\n')
