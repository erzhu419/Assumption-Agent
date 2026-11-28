import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
funcs={'sin':lambda x,r:np.sin(r*x),'tanh':lambda x,r:np.tanh(r*x),'erf':lambda x,r:__import__('scipy').special.erf(r*x),'oddexp':lambda x,r:-np.sign(x)*np.expm1(-r*abs(x)),'softsign':lambda x,r:x/(1+r*abs(x)),'sqrtodd':lambda x,r:x/np.sqrt(1+r*x*x),'rationalodd':lambda x,r:x/(1+r*x*x),'sqrteven':lambda x,r:x*x/np.sqrt(1+r*x*x),'rationaleven':lambda x,r:x*x/(1+r*x*x),'cos':lambda x,r:np.cos(r*x),'gauss':lambda x,r:np.exp(-r*x*x),'exp':lambda x,r:np.exp(r*x),'log':lambda x,r:np.log1p(r*x*x),'sinh':lambda x,r:np.sinh(r*x),'atan':lambda x,r:np.arctan(r*x),'softplus':lambda x,r:np.logaddexp(0,r*x)}
def des(u,k,r,extra):
 z=funcs[k](u,r)
 blocks=[np.ones(len(u)),z];
 if extra=='L':blocks.append(u)
 if extra=='Q':blocks.extend([u,*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]])
 return np.column_stack(blocks)
if __name__=='__main__':
 for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[p]['u'],s[p][k]; attempts=[]
  for key in funcs:
   for extra in ('','L','Q'):
    def obj(r):
     g=des(u,key,r,extra); return np.mean((g@ls(g,v)-v)**2)
    grid=np.linspace(.075,5,35); j=np.argmin([obj(r) for r in grid]); opt=minimize_scalar(obj,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded'); g=des(u,key,opt.x,extra); h=np.sum(g*np.linalg.pinv(g).T,axis=1); e=v-g@ls(g,v); cv=np.sqrt(np.mean((e/(1-h[:,None]))**2));attempts.append((cv,np.sqrt(opt.fun),key,extra,opt.x))
  print(p,k,*sorted(attempts)[:16],sep='\n')
 t=s['eval_times']; y=s['train']['traj'];
 for k,f in {'exp/exp':lambda a,b:np.array([-np.expm1(-a*t),-np.expm1(-b*t)]).T,'exp/linear':lambda a,b:np.array([-np.expm1(-a*t),t]).T,'exp/sin':lambda a,b:np.array([-np.expm1(-a*t),np.sin(b*t)]).T,'exp/texp':lambda a,b:np.array([-np.expm1(-a*t),t*np.exp(-b*t)]).T,'exp/cos':lambda a,b:np.array([-np.expm1(-a*t),1-np.cos(b*t)]).T}.items():
  def obj(p):g=f(*p);return (g@ls(g,y.T)-y.T).ravel()
  opt=least_squares(obj,[.8,.3],bounds=(.001,5)); print(k,opt.x,np.std(opt.fun))
