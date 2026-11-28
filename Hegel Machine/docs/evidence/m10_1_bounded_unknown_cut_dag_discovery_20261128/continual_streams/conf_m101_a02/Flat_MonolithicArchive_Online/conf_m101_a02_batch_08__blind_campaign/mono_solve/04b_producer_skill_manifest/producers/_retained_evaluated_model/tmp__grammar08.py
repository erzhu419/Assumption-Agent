from campaign_probe import *
from itertools import combinations,product
z,b,c,d=(a[:n] for a in old._coords(s)); x=tr['x']; y=tr['y']; T=tr['traj']; a=(z*np.roll(b,1,axis=1)).sum(1); base=np.c_[aff(b),a];
fix={'b2':(b*np.roll(b,2,axis=1)).sum(1)}; bank={}
for ka,A in [('z',z),('b',b),('c',c),('d',d),('x',x)]:
 for kb,B in [('z',z),('b',b),('c',c),('d',d),('x',x)]:
  for k in range(4):
   p=A*np.roll(B,k,axis=1); base_name=ka+kb+str(k)
   for key,q in [('s',p.sum(1)),('q',(p*p).sum(1)),('ss',p.sum(1)**2),('a',abs(p).sum(1)),('r',(p/(1+.2*p*p)).sum(1)),('t',np.tanh(p).sum(1))]:bank[base_name+key]=q
 for k in range(4):
  p=A[:,k];
  for key,q in [('p',p),('q',p*p),('c',p**3),('r',p/(1+.2*p*p)),('t',np.tanh(1.7*p)),('e',np.exp(-1.7*p*p)),('s',np.sin(1.7*p))]:bank[ka+str(k)+key]=q
 for p in [.1,.2,.5,1,1.7]:
  bank[f'{ka}norm{p}']=np.sum(A*A/(1+p*A*A),1);bank[f'{ka}cos{p}']=np.cos(p*A).sum(1)
for key,p in {'a':a,'b2':fix['b2']}.items():
 for v in [.1,.2,.5,1,1.7]:
  bank[key+'rat'+str(v)]=p/(1+v*p*p);bank[key+'exp'+str(v)]=p*np.exp(-v*p*p);bank[key+'sin'+str(v)]=np.sin(v*p)
 for k in range(2,5):bank[key+str(k)]=p**k
F=np.c_[base,fix['b2']]; keys=list(bank); G=np.column_stack(list(bank.values()));
for cut in [52,56]:
 def report(H):
  w=fit(H[:cut],y[:cut]);return np.std(y[:cut]-H[:cut]@w),np.mean(abs(y[cut:]-H[cut:]@w)),w
 active=[]
 for step in range(5):
  H=np.c_[F,G[:,active]]; err,mae,w=report(H); r=y[:cut]-H[:cut]@w; R=G[:cut]-H[:cut]@fit(H[:cut],G[:cut]); norm=np.linalg.norm(R,axis=0); gain=np.abs(R.T@r)/np.maximum(norm,1e-12);gain[norm<.002]=0;gain[active]=0
  top=np.argsort(gain)[-8:][::-1]; print(cut,step,round(err,5),round(mae,5),[keys[i] for i in active],[(keys[i],*[round(v,5) for v in report(np.c_[H,G[:,i]])[:2]]) for i in top]);active.append(top[0]);
 for aa in ['dd2q','dx0q','dd2ss','b2rat0.2','b233','a2','a3','b2exp0.2']:
  if aa in bank:print('named',cut,aa,report(np.c_[F,bank[aa]])[:2])
# Compare time laws after fitting the known cyclic amplitude across evaluation times.
for cut in [52,56]:
 Q=np.c_[aff(b),z,c,d]; H=np.c_[Q,a]; C=fit(H[:cut],T[:cut]);
 for label,W in [('log',np.log1p(s['telemetry_vec'][0]*times)),('offset_log',np.c_[np.ones(len(times)),np.log1p(times)]),('sqrt',np.sqrt(times)),('frac',times/(1+times)**.5),('log_sqrt',np.log1p(np.sqrt(times))),('free',np.eye(len(times)))]:
  W=np.asarray(W).reshape(len(times),-1); V=W@fit(W,C[-1]); print('time',cut,label,np.max(abs(V-C[-1])),np.mean(abs(np.c_[Q,a][cut:]@np.vstack([C[:-1],V])-T[cut:])))
