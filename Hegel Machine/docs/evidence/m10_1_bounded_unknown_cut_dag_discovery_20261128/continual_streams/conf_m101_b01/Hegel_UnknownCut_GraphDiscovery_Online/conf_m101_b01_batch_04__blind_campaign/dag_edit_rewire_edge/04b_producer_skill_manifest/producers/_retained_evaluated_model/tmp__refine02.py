import pickle, numpy as np
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
def load(i):
 ns={};exec(open(f'/holdout_task/_frozen_nodes/node_0{i}.py').read(),ns);return ns['run_node']
a=load(0)({'__input__':s}); u=a['calibrated_x']; z=load(1)({'node_00':a})['x']; n=len(s['train']['y']); Y=np.column_stack([s['train']['y'],s['train']['traj']]); t=s['eval_times']
def poly(x,p):
 d=x.shape[1];return np.column_stack([np.prod(x[:,k],axis=1) for j in range(p+1) for k in combinations_with_replacement(range(d),j)])
def loo(A,Y):
 c,_,rank,_=np.linalg.lstsq(A,Y,rcond=1e-11); h=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:rank]**2,axis=1); e=A@c-Y; return np.mean(np.abs(e),axis=0),np.mean(np.abs(e/np.maximum(1e-9,1-h[:,None])),axis=0)
p=s['panel_c']; q=p['u']; P=s['panel_b']; B=np.linalg.lstsq(poly(P['u'],1),P['v'],rcond=None)[0]; w=poly(q,1)@B
from scipy.linalg import lstsq
I=np.arange(n); folds=[(np.setdiff1d(I,h),h) for h in [I[-20:],I[:20],I[20:40],I[40:60]]]
def cv(A,reg=0,YY=Y):
 out=[]
 for j,k in folds:
  D=A[j]; V=A[k]; a=D.mean(0); scale=np.maximum(D.std(0),1e-10); D=(D-a)/scale; V=(V-a)/scale
  U,S,G=np.linalg.svd(D,full_matrices=False); c=(G.T*(S/(S**2+max(reg,1e-12))))@(U.T@(YY[j]-YY[j].mean(0))); err=np.mean(abs(V@c+YY[j].mean(0)-YY[k]),axis=0);out.append([err[0],err[1:].mean()]);
 return np.array(out)
from scipy.optimize import least_squares
G=np.c_[np.ones(len(q)),q]; GZ=np.c_[np.ones(len(z)),z]; GU=np.c_[np.ones(len(u)),u]
models={}
for name,g in [('tanh',np.tanh),('rat',lambda a:a/(1+a*a)),('sin',np.sin)]:
 def design(x,f):return np.c_[poly(x,1),g(x*f)]
 def res(f):
  d=design(q,f);return (d@np.linalg.lstsq(d,p['v1'],rcond=None)[0]-p['v1']).ravel()
 opt=min( [least_squares(res,np.full(4,start),bounds=(.15,8)) for start in [1.,1.7,2.2,2.8,3.4,4.]],key=lambda m:np.linalg.norm(m.fun)); f=opt.x; d=design(q,f); C=np.linalg.lstsq(d,p['v1'],rcond=None)[0];print('v1',name,f,'mae',np.mean(abs(opt.fun)),'C',C.T,flush=True);models[name]=(design(z,f)@C,design(u,f)@C,g(z*f))
for name,b in [('id',p['v2']),('atanh',np.arctanh(np.clip(p['v2'],-.999,.999)))]:
 C=np.linalg.lstsq(poly(q,2),b,rcond=None)[0]; P=poly(z,2)@C;print('v2',name,'mae',np.mean(abs(poly(q,2)@C-b)));models['v2'+name]=(P,poly(u,2)@C,P)
def pair(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def cv2(A,r):return cv(A[:n],r)
results=[]
for name,(v,w,h) in models.items():
 for inp,H in [('z',v),('u',w),('bare',h)]:
  for u_name,U in [('u',u),('z',z),('none',None)]:
   for form,K in [('linear',H),('quadratic',poly(H,2)),('squash',np.c_[H,np.tanh(H),np.sin(H)]),('bare+linear',np.c_[H,h])]:
    D=np.c_[z,K] if U is None else np.c_[z,K,pair(U,H)];
    for r in [.0001,.01,.1,1.]:
     E=cv2(D,r);results.append((max(E[:,0]),max(E[:,1]),name,inp,u_name,form,r,E[:,0],E[:,1]))
print('best raw scalar');
for R in sorted(results,key=lambda a:a[0])[:14]:print(R)
print('best traj');
for R in sorted(results,key=lambda a:a[1])[:8]:print(R)
