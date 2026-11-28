import pickle,numpy as np,itertools
from scipy.optimize import least_squares, minimize_scalar
from numpy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; LS=lambda d,y:lstsq(d,y,rcond=1e-10)[0]
def polys(x,k):
 inds=[j for d in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.column_stack([np.prod(x[:,j],axis=1) for j in inds]),inds
for panel,key in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 x,y=s[panel]['u'],s[panel][key]; keep=np.ones(len(x),bool)
 if panel=='panel_a':
  for _ in range(8):
   d=polys(x,1)[0];e=np.linalg.norm(d@LS(d[keep],y[keep])-y,axis=1);keep=e<4*np.median(e)
 x,y=x[keep],y[keep]; print('\n',panel,key)
 for deg in [1,2,3]:
  d,ix=polys(x,deg);w=LS(d,y);print('poly',deg,'rmse',np.std(d@w-y,axis=0),'largest terms',[(ix[j],np.round(w[j],3).tolist()) for j in np.argsort(np.linalg.norm(w,axis=1))[-8:]])
 d=polys(x,1)[0]
 funcs={'asinh':lambda z,k:np.arcsinh(k*z)/k,'cubic_inv':lambda z,k:2/np.sqrt(3*k)*np.sinh(np.arcsinh(1.5*np.sqrt(3*k)*z)/3),'signed_pow':lambda z,k:np.sign(z)*abs(z)**k,'rat':lambda z,k:z/(1+k*z*z),'even':lambda z,k:z*z/(1+k*z*z)}
 for name,f in funcs.items():
  if name!='even':
   gs=[]
   for j in range(4):
    v=y[:,j]; init=LS(d,v)
    def res(p):
     z=f(v-p[1],p[0]);return (z-d@LS(d,z))/(np.std(z)+1e-15)
    opt=least_squares(res,[.4,0],bounds=([.001,-2],[10,2]),max_nfev=200);gs.append([*opt.x,np.std(opt.fun)])
   print('post',name,np.round(gs,5))
 for j in range(4):
  print('channel',j)
  candidates=[]
  for kind,fun in {'even':lambda z,k:z*z/(1+k*z*z),'abs':lambda z,k:z/(1+k*abs(z)),'sin':lambda z,k:np.sin(k*z),'tanh':lambda z,k:np.tanh(k*z),'cross':lambda z,k:(z[0]*z[1])/(1+k*(z[0]*z[1])**2)}.items():
   for ii in (list(itertools.combinations(range(4),2)) if kind=='cross' else [(i,) for i in range(4)]):
    for pair in ([0] if len(ii)==2 else [0]+[h for h in range(4) if h!=ii[0]]):
     for sign in ([1] if pair==0 else [-1,1]):
      z=x[:,ii].T if len(ii)==2 else x[:,ii[0]] if pair==0 else x[:,ii[0]]+sign*x[:,pair]
      def loss(k):
       a=np.c_[d,fun(z,k)];return np.std(a@LS(a,y[:,j])-y[:,j]);
      opt=minimize_scalar(loss,bounds=(.001,3),method='bounded');candidates.append((opt.fun,kind,ii,pair,sign,opt.x))
  print(sorted(candidates)[:3])
