import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(A,B):return np.linalg.lstsq(A,B,rcond=1e-11)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def pol(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in it.combinations_with_replacement(range(x.shape[1]),k)])
def coords(x):
 a=s['panel_a']; u,v=a['u'],a['v']; m=np.linalg.norm(u-v,axis=1)<2
 b=np.array([fit(aff(u[m,j]),v[m,j]) for j in range(4)])
 return (x-b[:,0])/b[:,1]
bases={
 'sin':lambda x,k: np.c_[aff(x),np.sin(k*x),np.cos(k*x)],
 'odd_sin':lambda x,k: np.c_[aff(x),np.sin(k*x)],
 'pure_sin':lambda x,k: np.c_[np.ones(len(x)),np.sin(k*x),np.cos(k*x)],
 'tanh':lambda x,k: np.c_[aff(x),np.tanh(k*x),np.tanh(k*x)**2],
 'pure_tanh':lambda x,k: np.c_[np.ones(len(x)),np.tanh(k*x),np.tanh(k*x)**2],
 'rational':lambda x,k: np.c_[aff(x),x/(1+(k*x)**2),x*x/(1+(k*x)**2)],
 'sin_quad':lambda x,k: np.c_[aff(x),np.sin(k*x),x*x],
 'cos_quad':lambda x,k: np.c_[aff(x),np.cos(k*x),x**3],
 'shift_tanh':lambda x,k:np.c_[aff(x),np.tanh(k[:4]*x+k[4:])],
}
def operator(u,v,f,k,m=None):
 if m is None:m=np.ones(len(u),bool)
 A=f(u[m],k); b=fit(A,v[m]); return lambda x:f(x,k)@b

def learn(u,v,name,free=False,m=None):
 f=bases[name]; k0=np.ones(4 if free else 1)*1.5
 if name=='shift_tanh':k0=np.r_[np.ones(4)*1.5,np.zeros(4)]
 if m is None:m=np.ones(len(u),bool)
 def res(k):return (operator(u,v,f,k,m)(u[m])-v[m]).ravel()
 bnds=(.15,4.5) if name!='shift_tanh' else (np.r_[np.ones(4)*.15,[-3]*4],np.r_[[4.5]*4,[3]*4])
 k=least_squares(res,k0,bounds=bnds,max_nfev=100).x
 return k,operator(u,v,f,k,m)
if __name__=='__main__':
 for pan,key in [('b','v'),('c','v1'),('c','v2')]:
  p=s['panel_'+pan]; u,v=p['u'],p[key]; print(pan,key,flush=True)
  for name in bases:
   k,P=learn(u,v,name); ee=[]
   for i in range(len(u)):
    ki,Pi=learn(u,v,name,m=np.arange(len(u))!=i); ee.append(Pi(u[i:i+1])[0]-v[i])
   print(name,np.round(k,4),np.sqrt(np.mean((P(u)-v)**2)),np.sqrt(np.mean(np.square(ee))),flush=True)
  for name in ('sin','pure_sin','sin_quad','tanh','pure_tanh','rational'):
   k,P=learn(u,v,name,True); print('free',name,np.round(k,4),np.sqrt(np.mean((P(u)-v)**2)))
