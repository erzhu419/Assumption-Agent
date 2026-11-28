from cal_sat import *
from scipy.interpolate import PchipInterpolator

def selfpred(b,x,f):
 if f=='power': return b[0]+b[1]*x/(1+b[2]*x*x)**b[3]+b[4]*x
 if f=='pade': return b[0]+x*(b[1]+b[3]*x*x)/(1+b[2]*x*x+b[4]*x**4)
 return b[0]+b[1]*phi(b[2]*x,f)+b[3]*x

families=('tanh','atan','sqrt','rational','erf','power','pade')
def self_fit(x,y,f):
 if f=='power': b=[0,1.3,1.,.7,.1]; lo=[-1,0,0,0,-3]; hi=[1,10,10,6,3]
 elif f=='pade': b=[0,1.5,1.,.1,.02]; lo=[-1,0,0,-3,0]; hi=[1,10,20,10,10]
 else: b=[0,1.,1.,.1]; lo=[-1,0,.01,-3]; hi=[1,10,10,3]
 return least_squares(lambda b:selfpred(b,x,f)-y,b,bounds=(lo,hi),loss='soft_l1',f_scale=.001,max_nfev=450).x

def cross(x,k,form):
 u=x[:,k]; v=x[:,(k+2)%4]; a=x[:,(k+1)%4]; c=x[:,(k+3)%4]
 base=[v,a,c,v*v,u*v,u*a,u*c];
 if form=='sparse': return np.column_stack(base)
 if form=='opp': return np.column_stack(base+[v**3,v**4,v**5,u*v*v,u*u*v,u*u*v*v,v*a,v*c,v*a*c])
 return np.column_stack(base+[v**3,v**4,v**5,u*v*v,u*u*v,u*u*v*v,v*a,v*c,v*a*c,a*a,c*c,a*c,a**3,c**3,a*a*v,c*c*v,u*a*c])

if __name__=='__main__':
 for k in range(4):
  iz=(ids==k)&(abs(z)<5); u=sig[iz]; y=z[iz]; iz=np.flatnonzero(abs(Y[:,k])<5); x=X[iz]; r=Y[iz,k]; print('channel',k)
  for f in families:
   b=self_fit(u,y,f); e=np.array([selfpred(self_fit(u[np.arange(len(u))!=j],y[np.arange(len(u))!=j],f),u[j],f)-y[j] for j in range(len(u))]); print(f,'isolated loo max/central rmse',np.round([max(abs(e)),np.sqrt(np.mean(e[1:-1]**2))],5),'b',np.round(b,5))
   for form in ('sparse','opp','full'):
    D=cross(x,k,form); target=r-selfpred(b,x[:,k],f); q=np.empty(len(r));
    def clf(d,t):
     q=np.linalg.lstsq(d,t,rcond=None)[0]; return least_squares(lambda q:d@q-t,q,jac=lambda q:d.copy(),loss='soft_l1',f_scale=.002).x
    for j in range(5):
     tr=iz%5!=j; q[~tr]=D[~tr]@clf(D[tr],target[tr])-target[~tr]
    fit=clf(D,target); print(form,np.round(np.quantile(abs(q),[.5,.9,1]),5),np.round(fit,4) if f in ('power','pade') and form=='opp' else '')
