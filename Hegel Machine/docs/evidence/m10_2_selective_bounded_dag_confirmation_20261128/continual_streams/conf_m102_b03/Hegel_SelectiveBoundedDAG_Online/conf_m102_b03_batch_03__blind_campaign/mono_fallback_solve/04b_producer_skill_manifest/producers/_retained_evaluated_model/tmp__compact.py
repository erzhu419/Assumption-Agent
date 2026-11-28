from studyfit import *
f=np.load('/tmp/features.npz');z,L,C,D,Y,H=(f[k] for k in ('z','L','C','D','Y','H'));Y=Y[:,[0,9,10]]; N=52
print('time grid',s['eval_times']); print('target std',Y.std(0),'corr',np.corrcoef(Y.T));
for name,X in [('z',z),('L',L),('C',C),('D',D),('C+D',C+D),('C-D',C-D),('C+L',C+L),('C-L',C-L),('D+L',D+L),('D-L',D-L)]:
 bank=[]
 def add(n,v):bank.append((n,v))
 for j in range(4):
  for n,fun in [('id',lambda x:x),('sq',lambda x:x*x),('cube',lambda x:x**3),('tanh',np.tanh),('sin',np.sin),('rat',lambda x:x/(1+x*x))]:add(f'{n}{j}',fun(X[:,j]));
 for i,j in itertools.combinations(range(4),2):add(f'{i}*{j}',X[:,i]*X[:,j]);add(f'({i}-{j})^2',(X[:,i]-X[:,j])**2)
 for j in range(3):
  F=np.ones((len(X),1)); steps=[]
  B=np.array([v for _,v in bank]).T
  for _ in range(8):
   r=Y[:N,j]-F[:N]@ls(F[:N],Y[:N,j]); A=B[:N]-F[:N]@ls(F[:N],B[:N]);k=np.argmax(abs(r@A)/np.maximum(np.linalg.norm(A,axis=0),1e-8));F=np.c_[F,B[:,k]];e=abs(F@ls(F[:N],Y[:N,j])-Y[:,j]);steps.append((bank[k][0],round(e[:N].mean(),3),round(e[N:].mean(),3)))
  print(name,j,steps)
# Trajectory: regress temporal amplitudes with direct compact quadratic fits in rotated coordinates.
M=one(np.c_[L,pair(L)]);b=ls(M[:N],Y[:N,2]); print('slow full poly L',b,abs(M@b-Y[:,2]).mean());
