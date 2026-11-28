from maps import *
from scipy.optimize import minimize_scalar
from itertools import combinations

def id(x,k):return x
families={'rat':lambda x,k:x/(1+k*x*x),'rat2':lambda x,k:x/np.sqrt(1+k*x*x),'exp':lambda x,k:x*np.exp(-k*x*x),'soft':lambda x,k:x/(1+k*abs(x)),'asinh':lambda x,k:np.arcsinh(k*x),'atan':lambda x,k:np.arctan(k*x),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'sqrat':lambda x,k:x*x/(1+k*x*x),'tanh2':lambda x,k:np.tanh(k*x)**2,'cos':lambda x,k:np.cos(k*x),'tanh3':lambda x,k:np.tanh(k*x)**3,'rat3':lambda x,k:x**3/(1+k*x*x)}
def scan(x,y,ix,it,augment=True):
 rows=[]
 for n,fn in families.items():
  for lin in ([False,True] if augment else [False]):
   def f(x,k):return bias(np.c_[x,fn(x,k)] if lin else fn(x,k))
   def err(k):
    a=f(x[ix],k);r=(a@ls(a,y[ix])-y[ix]);return np.mean(r*r)
   grid=np.geomspace(.025,9,80);i=np.argmin([err(k) for k in grid]);r=minimize_scalar(err,bounds=(grid[max(0,i-1)],grid[min(i+1,len(grid)-1)]),method='bounded');a=f(x[ix],r.x);c=ls(a,y[ix]);p=f(x[it],r.x)@c;rows.append((np.mean(abs(y[it]-p)),r.fun,n,lin,r.x,c))
 return sorted(rows,key=lambda x:x[0])
if __name__=='__main__':
 for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=s[pn];u=p['u'];v=p[vn];top=scan(u,v,np.arange(20),np.arange(20,24));print(pn,vn,[r[:5] for r in top[:7]]);print(np.round(top[0][-1],5))
 tr=s['train'];x=inv(tr['x'],calfit(s));y=tr['y'];T=tr['traj'];t=s['eval_times'];T0=T[:56]; _,_,V=np.linalg.svd(T0,full_matrices=False);A=T@V[0]
 for nm,Y in [('scalar',y),('amp',A)]:
  top=scan(x,Y,np.arange(56),np.arange(56,72));print(nm,[r[:5] for r in top[:10]]);print('coeff',np.round(top[0][-1],4));f,n=poly(x,3);print('poly sparse');from probe import sparse;sparse(f,Y,n,5,8)
 print('rat gains',V[0]/V[0,-1]); print('rat path', t/(1+.46935*t)/(t[-1]/(1+.46935*t[-1])))
 for name,fun in [('rat_pow',lambda p:t/(1+p[0]*t)**p[1]),('log',lambda p:np.log1p(p[0]*t)),('exp_rat',lambda p:-np.expm1(-p[0]*t)/(1+p[1]*t)),('exp_osc',lambda p:-np.expm1(-p[0]*t)+p[1]*np.sin(p[2]*t)),('rat_osc',lambda p:t/(1+p[0]*t)+p[1]*np.sin(p[2]*t))]:
  def err(p):
   q=fun(p)[:,None];return (q@ls(q,T0.T)-T0.T).ravel()
  dim=1 if name=='log' else (3 if 'osc' in name else 2);r=least_squares(err,[.5,.05,2][:dim],bounds=([.001,-.2,.01][:dim],[8,5,12][:dim]));print(name,r.x,np.std(r.fun))
