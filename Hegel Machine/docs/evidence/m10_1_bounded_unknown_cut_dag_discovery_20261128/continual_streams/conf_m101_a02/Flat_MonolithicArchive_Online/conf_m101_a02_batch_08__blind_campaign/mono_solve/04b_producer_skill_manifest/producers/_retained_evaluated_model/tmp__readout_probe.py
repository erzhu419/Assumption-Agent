from campaign_probe import *
from itertools import combinations
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def analyze(label,z,b,c,d):
 y=tr['y']; T=tr['traj']; S=np.linalg.svd(T,full_matrices=False); h=S[2][0]; amp=T@h; M=np.c_[y,amp,T@S[2][1],T@S[2][2]]
 def score(L,l=52):
  return np.mean(abs(L[l:]@fit(L[:l],M[:l])-M[l:]),axis=0)
 blocks={'x':tr['x'],'z':z,'b':b,'c':c,'d':d,'all':np.c_[z,b,c,d]}; print('\n',label)
 for k,L in blocks.items(): print(k,'linear',score(aff(L)))
 for pair in [('z','b'),('z','c'),('b','c'),('z','d'),('b','d'),('c','d')]:
  a,g=[blocks[k] for k in pair]; print(pair,'lin',score(np.c_[aff(a),g]),'diag',score(np.c_[aff(a),g,a*g]),'full',score(np.c_[aff(a),g,(a[:,:,None]*g[:,None,:]).reshape(n,-1)]))
 X=np.c_[tr['x'],z,b,c,d]; names=[k+str(j) for k in ('x','z','b','c','d') for j in range(4)]; F={k:x for k,x in zip(names,X.T)}
 for i,a in enumerate(names):
  x=F[a]; F['tanh '+a]=np.tanh(x); F['sin '+a]=np.sin(x); F['exp '+a]=np.expm1(-1.7*x*x); F['cube '+a]=x**3; F['rational '+a]=x/(1+.2*x*x)
  for b in names[i:]: F[a+'*'+b]=x*F[b]
 FN=list(F); A=np.column_stack(list(F.values())); A=(A-A[:52].mean(0))/A[:52].std(0)
 for j in range(4):
  for base in [[],list(range(4,8)),list(range(8,12))]:
   ix=base.copy(); target=M[:,j]; print('target/base',j,base)
   for step in range(11):
    L=aff(A[:,ix]); proj=L[:52]@fit(L[:52],A[:52]); G=A[:52]-proj; r=target[:52]-L[:52]@fit(L[:52],target[:52]); metric=abs(G.T@r)/np.maximum(1e-8,np.linalg.norm(G,axis=0)); metric[ix]=0
    if step in (0,2,4,6,10):print(step,round(np.std(r),5),round(np.mean(abs(L[52:]@fit(L[:52],target[:52])-target[52:])),5),[FN[i] for i in ix])
    ix.append(np.argmax(metric))
z0,b0,c0,d0=old._coords(s); z,b,c,d=coords(); analyze('archive coordinates',*[q[:n] for q in (z0,b0,c0,d0)]); analyze('measured calibration',*[q[:n] for q in (z,b,c,d)])
U,S,V=np.linalg.svd(tr['traj'],full_matrices=False); print('time modes'); print(V[:3]);
for name,f in [('log',lambda p:np.log1p(p*times)),('exp',lambda p:-np.expm1(-p*times)),('power',lambda p:times**p),('sqrt',lambda p:np.sqrt(1+p*times)-1)]:
 for size in [1,2]:
  def fun(p):
   F=np.column_stack([f(p[0])]+([np.ones_like(times)] if size==2 else [])); return np.linalg.norm(V[0]-F@fit(F,V[0]));
  q=minimize_scalar(fun,bounds=(.001,15),method='bounded'); print(name,size,q.x,q.fun)
