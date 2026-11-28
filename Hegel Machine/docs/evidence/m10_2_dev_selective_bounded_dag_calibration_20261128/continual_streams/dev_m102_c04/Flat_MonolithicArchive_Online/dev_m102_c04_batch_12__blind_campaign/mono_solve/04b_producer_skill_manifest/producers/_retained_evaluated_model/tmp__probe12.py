import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(*a):return np.column_stack((np.ones(len(a[0])),*a))
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
u,v=[s['panel_a'][k] for k in ['u','v']]; m=np.max(abs(u-v),axis=1)<3
print('calibration differences sorted',np.sort(np.max(abs(u-v),axis=1)),flush=True)
for typ in ['diagonal','full','nonlinear']:
 for shift in (range(4) if typ=='nonlinear' else [0]):
  for xform in (['sin','tanh','cos'] if typ=='nonlinear' else ['sin']):
   a,b=u[m],v[m]; eye=np.broadcast_to(np.eye(4),(len(a),4,4)); base=[eye,(eye*a[:,:,None]) if typ!='full' else np.einsum('nij,nk->nijk',eye,a).reshape(len(a),4,16)]
   def mat(p):
    cols=base if typ!='nonlinear' else [*base,np.tanh(p[0]*a)[:,:,None],getattr(np,xform)(p[1]*np.roll(a,shift,axis=1))[:,:,None]]
    return np.concatenate(cols,axis=2).reshape(len(a)*4,-1)
   def res(p):
    f=mat(p); return f@ls(f,b.ravel())-b.ravel()
   k=least_squares(res,[1.8,2.4],bounds=([.1,.1],[6,6]),loss='soft_l1',f_scale=.04).x if typ=='nonlinear' else [0,0]; print('A',typ,shift,xform,k,np.quantile(abs(res(k)),[.5,.9,1]))
p=s['panel_c']; u=p['u']; pairs=list(itertools.combinations(range(4),2)); P=np.column_stack([u[:,i]*u[:,j] for i,j in pairs]);
for key in ['v1','v2']:
 v=p[key]; print('C',key)
 for label,f in [('pairs',D(P)),('allq',D(u,u*u,P)),('pureq',D(u*u,P)),('u',D(u))]: print(label,np.sqrt(np.mean((f@ls(f,v)-v)**2)))
 for label,fn in [('ratabs',lambda x,k:x/(1+k*abs(x))),('rat2',lambda x,k:x/(1+k*x*x)),('log',lambda x,k:np.log1p(k*x*x)),('sin',lambda x,k:np.sin(k*x)),('cos',lambda x,k:np.cos(k*x)),('tanh',lambda x,k:np.tanh(k*x)),('wave',lambda x,k:np.sin(k*x)+np.cos(k*x))]:
  for mixed in [False,True]:
   def res(k):
    f=D(u,fn(u,k)) if mixed else D(fn(u,k));return (f@ls(f,v)-v).ravel()
   out=min([least_squares(res,[k],bounds=(.005,8)) for k in [.4,1.2,2.5,4.5]],key=lambda q:q.cost); print(label,mixed,out.x,np.sqrt(np.mean(out.fun**2)))
