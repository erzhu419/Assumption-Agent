import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def A(x): return np.c_[np.ones(len(x)),x]
def fit(a,y): return np.linalg.lstsq(a,y,rcond=None)[0]
def poly(z,d): return np.column_stack([np.prod(z[:,t],axis=1) for k in range(d+1) for t in combinations_with_replacement(range(z.shape[1]),k)])
def score(f):
 e=Y-f@fit(f[:52],Y[:52]); return np.r_[np.abs(e[52:,0]).mean(),np.abs(e[52:,1:]).mean(),np.abs(e[:52,0]).mean(),np.abs(e[:52,1:]).mean()]
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.arange(len(u));
for _ in range(3):
 c=fit(A(u[keep]),v[keep]); e=np.linalg.norm(v-A(u)@c,axis=1); keep=np.argsort(e)[:45]
print('A matrix',c,'A sorted residuals',np.sort(e)); print('A quadratic rms',np.sqrt(np.mean((v[keep]-poly(u[keep],2)@fit(poly(u[keep],2),v[keep]))**2)))
b=s['panel_b']; cb=fit(A(b['u']),b['v']); pc=s['panel_c']; u=pc['u']; v=pc['v1'];
for name,f in [('self squash',A(np.tanh(u))),('sin',A(np.sin(u))),('tanh add',np.c_[A(u),np.tanh(u)]),('sin add',np.c_[A(u),np.sin(u)]),('cubic add',np.c_[A(u),u**3]),('quadratic',poly(u,2))]:
 print('C1',name,np.sqrt(np.mean((v-f@fit(f,v))**2)))
for name,f in [('diagonal',np.c_[A(u),u*u]),('cross',np.c_[A(u),*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]]),('quadratic',poly(u,2))]:
 print('C2',name,np.sqrt(np.mean((pc['v2']-f@fit(f,pc['v2']))**2)))
ci=fit(poly(u,2),pc['v2']); c1=fit(np.c_[A(u),np.tanh(u)],v)
for name,z in [('x',x),('ainv',(x-c[0])@np.linalg.inv(c[1:])),('a',A(x)@c)]:
 for suffix,w in [('',z),('b',A(z)@cb),('binv',(z-cb[0])@np.linalg.inv(cb[1:]))]:
  print(name+suffix,'affine',score(A(w)), 'deg2',score(poly(w,2)),'deg3',score(poly(w,3)))
  for key,g in [('tanh',np.tanh(w)),('sin',np.sin(w)),('C1',np.c_[A(w),np.tanh(w)]@c1),('C2',poly(w,2)@ci)]: print('  ',key,score(A(g)),score(np.c_[A(w),g]))
  if name+suffix=='ainvb':
   pp=poly(w,2); print('quadratic coeffs y, trajectory first/final',fit(pp[:52],Y[:52])[:,[0,1,-1]])
