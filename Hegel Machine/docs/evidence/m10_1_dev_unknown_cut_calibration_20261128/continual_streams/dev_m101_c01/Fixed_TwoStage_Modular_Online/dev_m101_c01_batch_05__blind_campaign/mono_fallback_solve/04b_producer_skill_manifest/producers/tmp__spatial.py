import numpy as np,pickle
from itertools import combinations_with_replacement as cw
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=(s['train'][k] for k in ('x','y','traj'))
def phi(x,d): return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in cw(range(4),k)])
def L(X,Y): return lstsq(X,Y,lapack_driver='gelsy')[0]
def affine(p):
 u,v=s[p]['u'],s[p]['v']; X=phi(u,1); I=np.arange(len(u))
 for j in range(7 if p=='panel_a' else 1):
  a=L(X[I],v[I]); I=np.argsort(np.linalg.norm(v-X@a,axis=1))[:36]
 if p=='panel_a':
  I=np.linalg.norm(v-X@a,axis=1)<.01
  a=np.vstack([np.zeros(4),np.eye(4)])
  for j in range(4): a[[0,j+1],j]=L(X[I][:,[0,j+1]],v[I,j])
 return a
A,B=affine('panel_a'),affine('panel_b')
def coords(x):
 a=(x-A[0])/A[1:].diagonal(); b=lambda v: phi(v,1)@B
 return {'raw':x,'a':a,'b':b(x),'ab':b(a),'ba':(b(x)-A[0])/A[1:].diagonal(),'bi':(x-B[0])@np.linalg.inv(B[1:])}
if __name__=='__main__':
 u=s['panel_c']['u']; D=phi(u,2); DT=D[:,[0,1,2,3,4,5,9,12,14]]
 for k in ['v1','v2']:
  v=s['panel_c'][k]; print(k)
  for nm,fun,lo,hi in [('sin',np.sin,.1,4),('tanh',np.tanh,.1,4),('cos',np.cos,.1,4),('sat',lambda w:w/(1+w*w),.1,4),('exp',np.exp,.1,3)]:
   for base in [D[:,:1],D[:,:5],DT]:
    def err(r):
     X=np.c_[base,fun(r*u)]; H=X@np.linalg.pinv(X); return np.mean(((v-H@v)/(1-np.diag(H))[:,None])**2)**.5
    opt=minimize_scalar(err,bounds=(lo,hi),method='bounded'); print(nm,base.shape[1],round(opt.x,4),round(opt.fun,6))
 def run(xx,mode,I,J,Y):
  X=phi(xx,2); Q=X[:,1:5]; n=len(I); scale=Y[I].std(0); Y=Y/scale
  if mode=='sine' or mode=='tanh':
   fn=np.sin if mode=='sine' else np.tanh
   def mat(p): return np.c_[X,fn(p[0]+Q@p[1:])]
   starts=[np.r_[ph,d*np.eye(4)[k]] for ph in [0.,1.] for d in [.8,2.] for k in range(4)]
  elif mode=='diagsat':
   def mat(p): return X/(1+(Q**2)@p)[:,None]
   starts=[np.full(4,.15)]; bounds=(0,np.inf)
  else:
   Z={'rad':np.c_[np.ones(len(Q)),Q,(Q**2).sum(1)],'lin':X[:,:5],'quad':X,'diag':X[:,[0,1,2,3,4,5,9,12,14]]}[mode]
   def mat(p): return X/(1+(Z@p)**2)[:,None]
   starts=[np.r_[ph,d*np.eye(4)[k],np.zeros(Z.shape[1]-5)] for ph in [0.,.5] for d in [.4,1.] for k in range(4)]
  def res(p):
   M=mat(p)[I]; R=M@L(M,Y[I])-Y[I]; return np.r_[R.ravel(),.002*p]
  opt=min((least_squares(res,p,bounds=bounds if mode=='diagsat' else (-5,5),max_nfev=65,ftol=1e-6) for p in starts),key=lambda r:r.cost)
  M=mat(opt.x); e=(M@L(M[I],Y[I])-Y)*scale
  return np.mean(abs(e[I]),axis=0),np.mean(abs(e[J]),axis=0),opt.x
 for i,(I,J) in enumerate([(np.arange(56),np.arange(56,72)),(np.arange(16,72),np.arange(16))]):
  _,_,V=np.linalg.svd(z[I],full_matrices=False); U=V[:2].T; Y=np.c_[y,z@U]; print('fold',i)
  for nm,xx in coords(x).items():
   for mode in ['lin','rad','diag','quad','diagsat','sine','tanh']:
    for j,yy in enumerate([Y[:,:1],Y[:,1:]]):
     a,b,p=run(xx,mode,I,J,yy); print(nm,mode,j,round(a.mean(),5),round(b.mean(),5),np.round(p,2) if i==0 else '',flush=True)
