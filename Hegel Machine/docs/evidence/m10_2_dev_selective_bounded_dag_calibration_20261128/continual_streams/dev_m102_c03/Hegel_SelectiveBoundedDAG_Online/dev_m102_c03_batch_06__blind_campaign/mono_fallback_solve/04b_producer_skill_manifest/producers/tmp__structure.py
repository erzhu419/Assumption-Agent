import pickle,numpy as np,importlib.util
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train'];t=s['eval_times']; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]; one=lambda x:np.c_[np.ones(len(x)),x]; rm=lambda e:np.sqrt(np.mean(e**2)); np.set_printoptions(precision=5,suppress=True,linewidth=160)
for k in ('a','b','c'):
 p=s['panel_'+k];u=p['u']; U=u.T; edge=u*np.roll(u,-1,axis=1); pair=np.array([U[i]*U[j] for i,j in combinations(range(4),2)]).T; trip=np.array([U[list(ij)].prod(0) for ij in combinations(range(4),3)]).T
 for key,v in p.items():
  if key=='u':continue
  good=np.linalg.norm(v-one(u)@ls(one(u),v),axis=1)<3 if k=='a' else np.ones(len(u),bool)
  def check(h):
   b=ls(h[good],v[good]); return rm((h@b-v)[good]),b
  for name,f in [('linear',u),('edge',edge),('pair',pair),('all quad',np.c_[u,pair,u*u]),('linear+cube',np.c_[u,u**3]),('linear+edge+cube',np.c_[u,edge,u**3]),('linear+edge+square+cube',np.c_[u,edge,u*u,u**3]),('pair+triple',np.c_[pair,trip]),('edge+triple',np.c_[edge,trip])]:
   e,b=check(one(f));print(k,key,name,e); 
   if k=='a' and name in ('linear+edge+cube','linear+edge+square+cube'):print(b)
  for name,fun in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('sat',lambda x:x/(1+abs(x))),('gauss',lambda x:np.exp(-x*x))]:
   for add in [np.empty((len(u),0)),edge,u]:
    f=lambda z:check(one(np.c_[fun(z*u),add]))[0]; grid=np.linspace(.15,8,101); j=np.argmin([f(z) for z in grid]); opt=minimize_scalar(f,bounds=(grid[max(0,j-1)],grid[min(100,j+1)]),method='bounded');
    if opt.fun<.32:print(k,key,name,'extra',len(add.T),opt.x,opt.fun)
