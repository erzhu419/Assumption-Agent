import numpy as np, pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def load(path):
 d={};exec(open('/holdout_task/'+path+'.py').read(),d);return d['run_node']
a=load('_frozen_nodes/node_00')({'__input__':s}); y=s['train']['y']; n=len(y); u=a['calibrated_x'][:n];z=load('_frozen_nodes/node_01')({'node_00':a})['x'][:n]; I=np.arange(n); folds=[(np.setdiff1d(I,k),k) for k in [I[-20:],I[:20],I[20:40],I[40:60]]];print('range',y.min(),y.max(),flush=True)

def Q(x):
 i,j=np.triu_indices(x.shape[1]);return x[:,i]*x[:,j]
def P(x,h):return (x[:,:,None]*h[:,None,:]).reshape(len(x),-1)
def specs(u,z,f):
 h=np.tanh(f*z); g=np.tanh(f*u); diag=np.c_[u*h,u*z]; K=np.c_[z,h]; return {'diag':np.c_[K,Q(h),diag], 'direct':np.c_[K,Q(h),u*h], 'pair':np.c_[K,Q(h),P(u,h)], 'mix':np.c_[K,Q(h),g*h], 'mix2':np.c_[K,Q(h),g*h,u*h], 'linmix':np.c_[K,Q(h),g], 'core':np.c_[K,Q(h)], 'mixedq':np.c_[K,Q(h),Q(u),u*h], 'sin':np.c_[K,Q(h),u*np.sin(f*z)], 'diagonly':np.c_[K,h*h,diag], 'crosspoly':np.c_[K,Q(h),Q(np.c_[u,z])], 'zpair':np.c_[K,Q(h),P(z,h)], 'cubh':np.c_[K,Q(h),u*h,h**3]}

def predict(D,y,E,reg,power=1.):
 A=np.c_[np.ones(len(D)),D]; E=np.c_[np.ones(len(E)),E]; scale=np.maximum(np.sqrt((A*A).mean(axis=0)),1e-9); A=A/scale; E=E/scale
 U,S,V=np.linalg.svd(A,full_matrices=False); k=S>1e-7; U=U[:,k]; S=S[k]; V=V[k]; T=E@(V.T/S); ini=U.T@y
 def link(x):return x/(1+abs(x)**(2*power))**(.5/power)
 def res(a):
  b,m,o=a[:-2],a[-2],a[-1]; x=U@b; r=m*link(x)+o-y
  return np.r_[r,reg*(V.T@(b/S))[1:]] if reg else r
 def jac(a):
  x=U@a[:-2]; j=np.column_stack((a[-2]*((1+abs(x)**(2*power))**(-1-.5/power))[:,None]*U,link(x),np.ones(len(y))))
  return np.vstack((j,np.c_[reg*(V.T/S)[1:],np.zeros((A.shape[1]-1,2))])) if reg else j
 fit=min((least_squares(res,np.r_[ini/m,m,y.mean()],jac=jac,max_nfev=60) for m in (.5,1.,2.)),key=lambda f:f.cost); a=fit.x
 return a[-2]*link(T@a[:-2])+a[-1], np.mean(abs(res(a)[:len(y)]))

if __name__=='__main__':
 out=[]
 for f in (1.2,1.6,2.,2.2,2.5,3.,3.5,4.):
  for name,A in specs(u,z,f).items():
   for r in (.003,.015,.06,.16):
    ee=[];tt=[]
    for j,k in folds:
     yp,t=predict(A[j],y[j],A[k],r);ee.append(np.mean(abs(yp-y[k])));tt.append(t)
    out.append((max(ee),np.mean(ee),f,name,r,np.round(ee,5),np.mean(tt)))
 print('best robust',*sorted(out,key=lambda x:x[0])[:18],sep='\n'); print('best train',*sorted(out,key=lambda x:x[-1])[:8],sep='\n')
 for name in ('direct','diag','mix','linmix','core','zpair'):
  R=min((r for r in out if r[3]==name),key=lambda x:x[0]);print('per-family',R)
  for power in (.55,.75,1.25,1.75,2.5):
   f,r=R[2],R[4]; A=specs(u,z,f)[name]; ee=[np.mean(abs(predict(A[j],y[j],A[k],r,power)[0]-y[k])) for j,k in folds];print('link',name,f,r,power,np.round(ee,5),flush=True)
