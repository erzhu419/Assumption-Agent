exec(open('/tmp/probe.py').read().split('for name, aa')[0])
from itertools import combinations,product
v=vs[0]; M=ls(D(s['panel_b']['u']),s['panel_b']['v']); dirs=[]
def add(w,name):
 if not any(np.allclose(w,t[0]) for t in dirs): dirs.append((w,name))
I=np.eye(4)
for i in range(4):
 for j in range(4):
  if i==j: add(I[i],str(i)); continue
  for f in [-2,-1.25,-1,-.75,-.5,.5,.75,1,1.25,2]:add(I[i]+f*I[j],f'{i}+{f}*{j}')
for mat,name in [(M[1:],'B'),(M[1:].T,'Bt'),(np.linalg.inv(M[1:]),'Bi')]:
 for i in range(4):add(mat[:,i],name+str(i))
func={'soft':b._soft,'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'inv':lambda x:1/(1+x*x),'r':lambda x:x/(1+x*x),'cube':lambda x:x**3,'quad':lambda x:x*x,'abs':np.abs}
lib=[]; names=[]
for w,nm in dirs:
 for k,f in func.items():
  for f0 in ([.4,.84,1.25,1.7,2.15,2.4,3,4,5,6] if k in ['soft','tanh','sin','cos','inv','r'] else [1.]):
   lib.append(f(f0*(u@w)));names.append((nm,k,f0))
for i,j in combinations(range(4),2):
 for k,f in func.items():
  for t in [.84,1.25,2.15,2.4]:
   lib.extend([u[:,i]*f(t*u[:,j]),u[:,j]*f(t*u[:,i])]);names.extend([(i,j,k,t),(j,i,k,t)])
F=np.array(lib).T; Z=D(u); R=v-Z@ls(Z,v); F-=Z@ls(Z,F); F/=np.maximum(np.linalg.norm(F,axis=0),1e-12)
print('residual svd',np.linalg.svd(R,compute_uv=False));print('v1 ranges',v.min(0),v.max(0));print('linear',ls(Z,v));print('top',[(names[i],round(np.linalg.norm(F[:,i]@R),5)) for i in np.argsort(np.linalg.norm(F.T@R,axis=1))[-20:][::-1]])
for i in range(8):
 j=np.argmax(np.linalg.norm(F.T@R,axis=1)); q=F[:,j].copy();R-=np.outer(q,q@R); F-=np.outer(q,q@F); F/=np.maximum(np.linalg.norm(F,axis=0),1e-12);print(i,names[j],np.std(R))
print('v1',np.round(np.c_[u,v],3))
