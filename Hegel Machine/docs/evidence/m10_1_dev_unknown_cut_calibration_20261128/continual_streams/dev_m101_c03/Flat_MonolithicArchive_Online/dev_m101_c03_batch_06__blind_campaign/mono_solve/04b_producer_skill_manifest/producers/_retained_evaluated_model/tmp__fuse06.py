from spatial06 import *
z,w,((a,b),(aw,bw))=geom(s);tr=s['train'];n=len(tr['y']); t=s['eval_times'];S=tr['traj']; y=tr['y']; T=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(ls(T(p),S[:56].T).T@T(p).T-S[:56]).ravel(),[.4,1.8,5]).x; B=T(p);C=ls(B,S.T).T; Y=np.c_[y,C]; I=np.arange(56);J=np.arange(56,n); dot=(a*b).sum(1); ab=a[:,0]*b[:,1]; F=aff(np.c_[a,b,dot,ab]);R=Y-F[:n]@ls(F[I],Y[I]);print('residual correlation',np.round(np.corrcoef(R.T),4));print('mode regression coefficients',np.round(ls(F[I],Y[I]),4));print('y in C',ls(aff(C[I]),y[I]),np.mean(abs(y[J]-aff(C[J])@ls(aff(C[I]),y[I]))));print('resid eig',np.linalg.svd(R[I],compute_uv=False));
groups={'z':z,'w':w,'a':a,'b':b,'aw':aw,'bw':bw}; fs={'dot':dot,'ab':ab}
for key,V in groups.items():
 for i,x in enumerate(V.T):fs[key+str(i)]=x
 for i,j in combinations(range(4),2):
  fs[f'{key}{i}+{j}']=V[:,i]+V[:,j];fs[f'{key}{i}-{j}']=V[:,i]-V[:,j]
for X1,X2 in [('a','b'),('aw','bw'),('a','bw'),('aw','b')]:
 U,V=groups[X1],groups[X2];fs[f'{X1}dot{X2}']=np.sum(U*V,1)
 for i,j in np.ndindex(4,4):fs[f'{X1}{i}*{X2}{j}']=U[:,i]*V[:,j]
for key,x in list(fs.items()):
 for k,f in [('sq',lambda x:x*x),('cube',lambda x:x**3),('abs',np.abs),('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('rec',lambda x:x/(1+x*x)),('even',lambda x:x*x/(1+x*x))]:fs[k+'('+key+')']=f(x)
 for k in (.25,.5,2.,4.):fs[f'tanh{k}({key})']=np.tanh(k*x);fs[f'sin{k}({key})']=np.sin(k*x);fs[f'even{k}({key})']=x*x/(1+k*x*x)
for key,V in groups.items():
 for j in range(4):
  for name,x in list(fs.items())[:2]:
   for k in (.25,1,4):fs[f'{key}{j}/(1+{k}{name}²)']=V[:,j]/(1+k*x*x)
for key,V in groups.items():
 for i,j,k in combinations(range(4),3):fs[f'{key}{i}{j}{k}']=V[:,i]*V[:,j]*V[:,k]
D=np.array(list(fs.values())).T; names=np.array(list(fs));
for q,target in enumerate(Y.T):
 for base in [aff(z[:,:0]),F,aff(np.c_[a,b,z,w,ab,dot])]:
  H=base.copy(); chosen=[]
  for step in range(6):
   U=D-H@ls(H[I],D[I]); E=target-H[:n]@ls(H[I],target[I]);norm=np.linalg.norm(U[I],axis=0); corr=np.divide(U[I].T@E[I],norm,out=np.zeros(len(norm)),where=norm>1e-7);j=np.argmax(abs(corr));chosen.append(names[j]);H=np.c_[H,D[:,j]]
   if step in (0,2,5):
    e=target-H[:n]@ls(H[I],target[I]);print(q,base.shape[1],chosen,'rmse/mae',round(np.std(e[I]),5),round(np.mean(abs(e[J])),5))
