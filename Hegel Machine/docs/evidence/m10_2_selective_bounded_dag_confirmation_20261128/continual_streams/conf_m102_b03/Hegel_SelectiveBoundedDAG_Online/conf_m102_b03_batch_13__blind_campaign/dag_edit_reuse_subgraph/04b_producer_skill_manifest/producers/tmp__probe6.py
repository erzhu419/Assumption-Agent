import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v2']; n,d=u.shape
O=lambda x:np.c_[np.ones(len(x)),x]; LS=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
E=v-O(u)@LS(O(u),v); print('singular values raw/linear residual:',np.linalg.svd(v,compute_uv=False),np.linalg.svd(E,compute_uv=False)); print('u ranges',u.min(0),u.max(0)); print('linear coeffs',LS(O(u),v))
e=np.eye(4); prim={f'x{i}':u[:,i] for i in range(4)}; pairs=list(itertools.combinations(range(4),2));
for i,j in pairs:
 x,y=u[:,i],u[:,j]; prim[f'x{i}*x{j}']=x*y
 for k in [.5,1,1.5,2]:
  for sign in [-1,1]: prim[f'x{i}+{sign*k}x{j}']=x+sign*k*y
for inds in itertools.combinations(range(4),3):prim[str(inds)]=u[:,inds].prod(1)
D={}
for name,z in prim.items():
 D[name]=z; D[name+'^2']=z*z; D[name+'^3']=z**3; D['abs '+name]=abs(z)
 for k in [.15,.3,.5,.8,1,1.3,1.7,2,2.15,2.4,2.8,3.2,3.6,4,5]:
  for label,f in [('sin',np.sin(k*z)),('cos',np.cos(k*z)),('tanh',np.tanh(k*z)),('sat',z/(1+k*abs(z))),('rat',z/(1+k*z*z)),('rat2',z*z/(1+k*z*z))]:D[f'{label}({name};{k})']=f
 for j in range(4):
  for k in [.15,.3,.5,.8,1.25,2.15]: D[f'{name}/(1+{k}x{j}^2)']=z/(1+k*u[:,j]**2)
for i,j in pairs:
 x,y=u[:,i],u[:,j]
 for k in [.15,.3,.5,.8,1.25,2.15]:
  for sg in [-1,1]: D[f'x{i}x{j}/(1+{k}(x{i}{sg:+}x{j})^2)']=x*y/(1+k*(x+sg*y)**2)
A=np.column_stack(list(D.values())); names=np.array(list(D));
for init in [O(u),np.ones((n,1))]:
 for col in [-1,0,1,2,3]:
  V=v if col<0 else v[:,col:col+1]; F=init; out=[]
  for _ in range(6):
   Q=np.linalg.qr(F)[0]; r=V-Q@(Q.T@V); B=A-Q@(Q.T@A); norm=np.linalg.norm(B,axis=0); sc=np.linalg.norm(B.T@r,axis=1)/np.maximum(norm,1e-9); j=np.argmax(sc); out.append(names[j]); F=np.c_[F,A[:,j]]
   if _==0 and col==-1: print('first alternatives',list(zip(names[np.argsort(sc)[-12:][::-1]],np.sort(sc)[-12:][::-1])))
  print(init.shape[1],col,np.mean(abs(V-F@LS(F,V))),out)
# Screen modular output-row laws using linear + one nonlinear atom, allowing separate row bases.
for j in range(4):
 for col in range(4):
  F=O(u); Q=np.linalg.qr(F)[0]; B=A-Q@(Q.T@A); r=v[:,col]-Q@(Q.T@v[:,col]); sc=(B.T@r)**2/np.maximum((B*B).sum(0),1e-14); k=np.argsort(sc)[-5:][::-1]; print('single',col,names[k].tolist(),np.sqrt(np.maximum((r*r).sum()-sc[k],0)/n));
 break
