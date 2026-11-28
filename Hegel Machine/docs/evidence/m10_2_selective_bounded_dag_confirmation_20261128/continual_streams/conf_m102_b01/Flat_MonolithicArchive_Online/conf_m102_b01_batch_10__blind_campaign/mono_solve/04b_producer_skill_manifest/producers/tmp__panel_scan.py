import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; LS=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(u,d):return np.column_stack([np.prod(u[:,k],axis=1) for i in range(1,d+1) for k in itertools.combinations_with_replacement(range(4),i)])
def press(d,v):
 p=np.linalg.pinv(d);return np.sqrt(np.mean(((d@p@v-v)/np.maximum(.01,1-np.sum(d*p.T,axis=1))[:,None])**2))
a=s['panel_a'];u=a['u'];v=a['v']; m=np.linalg.norm(v,axis=1)<7;u=u[m];v=v[m];print('A inliers',len(u))
for j in range(4):
 x=u[:,j]; z=u[:,(j+2)%4];n=u[:,(j+1)%4];p=u[:,(j-1)%4];
 for typ,d in [('archive09',np.c_[x,np.tanh(1.8*x),np.sin(2.4*z)]),('poly',poly(u,2)),('cycle',np.c_[x,n,x*n,x**3]),('broad',np.c_[x,np.tanh(1.8*x),np.sin(2.4*x),z,np.sin(2.4*z),x*z,x**3,n,p])]:
  d=aff(d);print(j,typ,press(d,v[:,j:j+1]),LS(d,v[:,j]).round(4))
funcs={'linear':lambda x,a:x,'tanh':lambda x,a:np.tanh(a*x),'sin':lambda x,a:np.sin(a*x),'atan':lambda x,a:np.arctan(a*x),'asinh':lambda x,a:np.arcsinh(a*x),'inv3':lambda x,a:np.sinh(np.arcsinh(a*x)/3),'dead':lambda x,a:np.sign(x)*np.maximum(abs(x)-a,0)**1.5,'dead1':lambda x,a:np.sign(x)*np.maximum(abs(x)-a,0),'softsign':lambda x,a:x/(1+a*abs(x)),'sinhn':lambda x,a:np.sinh(a*x),'cubic':lambda x,a:x+a*x**3,'sat':lambda x,a:x/np.sqrt(1+a*x*x)}
for k,o in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 q=s[k];u=q['u'];v=q[o];out=[]
 for name,f in funcs.items():
  for name2,g in funcs.items():
   for j in [0,1]:
    def fit(a):
     x=f(u,a); y=g(v,a); return press(aff(np.c_[x,u] if j and name!='linear' else x),y)/np.std(y)
    z=minimize_scalar(fit,bounds=(.04,.75 if 'dead' in name+name2 else 4),method='bounded');out.append((z.fun,name,name2,j,z.x))
 cross=np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)]); print(k,o,'cross',press(aff(cross),v),'cross4',press(aff(cross[:,[0,2,3,5]]),v),'quad',press(aff(poly(u,2)),v),'top',sorted(out)[:20])
