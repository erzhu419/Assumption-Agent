from discover import *
from scipy.optimize import least_squares
p=s['panel_c']; U=p['u']; Q=lambda a:design(np.cos(2.4*a)-1); Wc=fit(Q(U),p['v2']); C=lambda a:Q(a)@Wc
b=lambda a:design(a)@B
D=lambda a: np.c_[b(a),v1(a),C(a)]
q=lambda r:-np.expm1(-t[:,None]*np.asarray(r))
r=np.array([.395,1.822,4.816]); A=np.c_[Y[:,0],fit(q(r),Y[:,1:].T).T];print('times',t);print('trajectory modal std',A.std(0));
T=[(f'z{i}',a) for i,a in enumerate(z.T)]+[(f'B{i}',a) for i,a in enumerate(b(z).T)]+[(f'P{i}',a) for i,a in enumerate(v1(z).T)]+[(f'C{i}',a) for i,a in enumerate(C(z).T)]+[(f'cosz{i}',a) for i,a in enumerate(np.cos(2.4*z).T)]
for label,M in [('B',b(z)),('P',v1(z)),('C',C(z))]:print(label,'correlations',np.corrcoef(M.T,A.T)[:4,4:].round(2));
for i in range(4):
 for j in range(i+1,4):T.extend([(f'z{i}+z{j}',z[:,i]+z[:,j]),(f'z{i}-z{j}',z[:,i]-z[:,j])]);
P=T.copy()
for a,u in T:
 for k in (.5,1,1.5,2,2.4,3,4,5,6,8,10):
  for f in (np.sin,np.cos,np.tanh):P.append((f'{f.__name__}{k}({a})',f(k*u)))
for (a,u),(b0,v) in it.combinations_with_replacement(T[:20],2):
 for expr,term in [(f'{a}*{b0}',u*v),(f'{a}/(1+{b0}^2)',u/(1+v*v)),(f'{b0}/(1+{a}^2)',v/(1+u*u)),(f'sin({a}+{b0})',np.sin(u+v)),(f'cos({a}-{b0})',np.cos(u-v))]:P.append((expr,term))
F=np.column_stack([v for _,v in P]); G=(F-F.mean(0))/np.maximum(F.std(0),1e-8); H=A-A.mean(0)
for j,y in enumerate(H.T):
 top=np.argsort(abs(G.T@y))[-8:][::-1]; print('corr',j,[(P[k][0],round(G[:,k]@y/(len(y)*y.std()),3)) for k in top]); ix=[];e=y.copy()
 for _ in range(10):
  k=np.argmax(abs(G.T@e));ix.append(k); w=fit(G[:,ix],y);e=y-G[:,ix]@w;print(j,P[k][0],round(np.std(e),4),end='; ')
 print()
for a,Z in [('z',z),('B',b(z)),('BPC',D(z)),('zBPC',np.c_[z,D(z)])]:
 for deg in [1,2,3]:
  ids2=[ix for k in range(1,deg+1) for ix in it.combinations_with_replacement(range(Z.shape[1]),k)]; M=poly(Z,ids2); print(a,deg,'fit rms',np.std(M@fit(np.c_[np.ones(len(M)),M],Y)[1:]-Y,axis=0).round(3));
  print(a,deg,'ridge',[(i,metrics(M,'ridge',i).mean(0).round(3)) for i in [.001,.1,1,10]])
