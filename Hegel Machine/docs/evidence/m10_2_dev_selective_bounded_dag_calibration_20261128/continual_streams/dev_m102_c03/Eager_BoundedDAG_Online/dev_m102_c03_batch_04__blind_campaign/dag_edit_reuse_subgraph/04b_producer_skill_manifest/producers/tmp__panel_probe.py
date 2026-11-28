exec(open('/tmp/fit_probe.py').read().split('R=[]')[0].split('for k in (')[0]); from itertools import product
np.set_printoptions(suppress=True,precision=4)
M=[]
for name in ['panel_a','panel_b']:
 p=s[name]; u=p['u']; v=p['v']; g=(abs(u-v).max(1)<1) if name=='panel_a' else np.ones(len(u),bool); a=np.linalg.lstsq(np.c_[np.ones(sum(g)),u[g]],v[g],rcond=None)[0]; M.append(a); print(name,a,'rms',np.std(v[g]-np.c_[np.ones(sum(g)),u[g]]@a,0))
A,B=M; p=s['panel_c']; u=p['u']; T=np.c_[p['v1'],p['v2']]; print('c target corr\n',np.corrcoef(T.T));
trans={'raw':u,'B':u@B[1:]+B[0],'BT':u@B[1:].T,'Bi':(u-B[0])@np.linalg.inv(B[1:]),'A':u@A[1:]+A[0],'Ai':(u-A[0])@np.linalg.inv(A[1:])};
for name,h in trans.items():
 R=[]
 for family,k in product(['sin','cos','tanh','even','abs','quad','sig','cube'],np.geomspace(.1,8,45)):
  f={'sin':lambda:np.sin(k*h),'cos':lambda:np.cos(k*h),'tanh':lambda:np.tanh(k*h),'even':lambda:np.exp(-k*h*h),'abs':lambda:abs(h),'quad':lambda:h*h,'sig':lambda:h/(1+k*abs(h)),'cube':lambda:h**3}[family]();
  F=np.c_[np.ones(len(h)),f]; G=np.c_[F,h];
  for tag,J in [('nonlinear',F),('pluslin',G)]:
   err=J@np.linalg.lstsq(J,T,rcond=None)[0]-T; R.append((tag,family,k,np.sqrt(np.mean(err[:,:4]**2)),np.sqrt(np.mean(err[:,4:]**2))))
 print(name,'v1',sorted(R,key=lambda r:r[3])[:4],'v2',sorted(R,key=lambda r:r[4])[:4]);
print('target corr',np.corrcoef(np.column_stack([Y[:,0],Y[:,1:]@np.linalg.pinv(Q).T]).T))
