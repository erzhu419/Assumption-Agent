from recover import *
np.set_printoptions(precision=5,suppress=True,linewidth=130)
for key,out,fun,w in [('panel_b','v',np.sin,2.887),('panel_b','v',np.tanh,1.4255),('panel_c','v1',np.tanh,2.1559),('panel_c','v2',np.cos,2.6077)]:
 u=s[key]['u']; v=s[key][out]; a=aff(u); A=ls(np.c_[a,fun(w*u)],v); print(key,out,fun.__name__,'relative mixing',A[5:]@np.linalg.inv(A[1:5]),'gram',A[1:5]@A[1:5].T,sep='\n')
 Z=np.linalg.svd(A[1:5])[0]@np.linalg.svd(A[1:5])[2]; V=(v-v.mean(0))@Z.T
 for j in range(4):
  ij=np.argsort(u[:,j]); print('xy',j,np.c_[u[ij,j],V[ij,j]][::3].T)
tr=s['train']['traj']; t=s['eval_times']; n=56
print('times',t)
for name,base in [('relax',lambda r:-np.expm1(-t[:,None]*np.exp(r))),('relax+initial',lambda r:np.exp(-t[:,None]*np.exp(r))),('relax+linear',lambda r:np.c_[-np.expm1(-t*np.exp(r[0])),t]),('relax+osc',lambda r:np.c_[-np.expm1(-t*np.exp(r[0])),np.sin(t*np.exp(r[1]))])]:
 def r(p):
  F=base(p);return (F@ls(F,tr[:n].T)-tr[:n].T).ravel()
 o=least_squares(r,np.log([.5,2]),bounds=(np.log(.03),np.log(15))); print(name,np.exp(o.x),np.std(o.fun))
z=coords(); Y=np.c_[s['train']['y'],tr];
for k in ['x','bs','bt','ct','dc','bs+ct']:
 x=z.get(k,z['bs']); des=np.c_[aff(x),cross(x)];
 if k=='bs+ct':des=np.c_[des,z['ct'],cross(z['ct'])]
 for f in [des,np.c_[des,z['x']]]:
  w=ls(f,Y); e=Y[n:]-f[n:]@ls(f[:n],Y[:n]);print(k,f.shape[1],'MAE',np.mean(abs(e),axis=0),'coeff y',w[:,0]);print('coeff terminal',w[:,-1])
