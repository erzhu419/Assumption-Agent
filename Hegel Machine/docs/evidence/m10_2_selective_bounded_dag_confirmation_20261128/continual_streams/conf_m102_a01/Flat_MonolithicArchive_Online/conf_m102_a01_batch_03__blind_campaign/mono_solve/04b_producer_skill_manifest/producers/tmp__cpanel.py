import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; np.set_printoptions(precision=4,suppress=True)
for col in ('v1','v2'):
 v=p[col]; print(col)
 for label,H in [('q',u*u),('sin',np.sin(u)),('cos',np.cos(u)),('tanh',np.tanh(u)),('q+c',np.c_[u*u,u**3]),('q+4',np.c_[u*u,u**4]),('q+c+4',np.c_[u*u,u**3,u**4]),('pair',np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]))]:
  A=np.c_[np.ones(len(u)),u,H]; C=np.linalg.pinv(A); E=v-A@(C@v); h=1-np.sum(A*C.T,axis=1); print(label,np.sqrt(np.mean(E**2)),np.sqrt(np.mean((E/h[:,None])**2)))
 for k in range(4):
  A=np.c_[np.ones(len(u)),u,u*u]; c=np.linalg.lstsq(A,v[:,k],rcond=None)[0];print(c)
