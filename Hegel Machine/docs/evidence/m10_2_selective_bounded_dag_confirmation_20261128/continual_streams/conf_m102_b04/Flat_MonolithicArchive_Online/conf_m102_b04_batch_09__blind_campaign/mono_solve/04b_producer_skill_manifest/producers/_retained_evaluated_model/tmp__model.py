import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(X,y):return np.linalg.lstsq(X,y,rcond=1e-10)[0]
def af(x):return np.c_[np.ones(len(x)),x]
def quad(x):return np.c_[af(x),*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)]]
def rm(e):return np.mean(e**2)**.5
def mae(e):return np.mean(abs(e),axis=0)
def funcs():return {'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'rat':lambda x,k:x/(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'sqrt':lambda x,k:x/np.sqrt(1+k*x*x),'soft':lambda x,k:x/(1+k*abs(x)),'sinh':lambda x,k:np.sinh(k*x),'exp':lambda x,k:x*np.exp(-k*x*x)}
pa=s['panel_a'];u,v=pa['u'],pa['v'];keep=np.linalg.norm(u-v,axis=1)<2;u,v=u[keep],v[keep];print('keep',sum(keep))
pc=s['panel_c'];c,v1,v2=[pc[k] for k in ['u','v1','v2']]
def calbasis(u,f,k,j,mode):
 a,b=u[:,j],u[:,(j+2)%4]; cols=[np.ones(len(u)),a,b,f(a,k)];
 if mode>=1:cols.extend([a*b,a*a]);
 if mode>=2:cols.extend([b*b,f(b,k)]);
 if mode>=3:cols.extend([u[:,(j+1)%4],u[:,(j+3)%4]]);
 return np.column_stack(cols)
def fitcal(f,mode):
 def mats(k):return [calbasis(u,f,k,j,mode) for j in range(4)]
 def res(p):return np.array([X@ls(X,v[:,j])-v[:,j] for j,X in enumerate(mats(p[0]))]).ravel()
 fit=min([least_squares(res,[k],bounds=(.035,5)) for k in [.2,.8,1.5,3.]],key=lambda x:x.cost);k=fit.x[0];w=np.array([ls(X,v[:,j]) for j,X in enumerate(mats(k))]);return rm(fit.fun),k,w
def calibrate(data,f,k,w,mode):
 def forward(x):return np.column_stack([calbasis(x,f,k,j,mode)@w[j] for j in range(4)])
 x=data.copy()
 for _ in range(15):
  r=forward(x)-data
  J=np.stack([(forward(x+e*1e-5)-forward(x-e*1e-5))/2e-5 for e in np.eye(4)],axis=-1)
  dx=np.linalg.solve(J,r[...,None])[...,0];x-=dx/np.maximum(1,np.max(abs(dx),axis=1,keepdims=True))
 return x
if __name__=='__main__':
 for n,f in funcs().items():
  fits=[fitcal(f,m) for m in range(4)];print('cal',n,[(round(r,6),round(k,4)) for r,k,w in fits]);
  if min(x[0] for x in fits)<.003:print(fits[0][2],fits[3][2])
 for target in [v1,v2]:
  for n,f in funcs().items():
   vals=[]
   for base in [lambda x:np.ones((len(x),1)),af,lambda x:np.c_[af(x),x*x],quad]:
    def res(p):
     X=np.c_[base(c),f(c,p[0])];return (X@ls(X,target)-target).ravel()
    a=min([least_squares(res,[k],bounds=(.05,5)) for k in [.2,1.,2.,4.]],key=lambda x:x.cost);vals.append((round(rm(a.fun),6),round(a.x[0],4)))
   print('C',n,vals)
