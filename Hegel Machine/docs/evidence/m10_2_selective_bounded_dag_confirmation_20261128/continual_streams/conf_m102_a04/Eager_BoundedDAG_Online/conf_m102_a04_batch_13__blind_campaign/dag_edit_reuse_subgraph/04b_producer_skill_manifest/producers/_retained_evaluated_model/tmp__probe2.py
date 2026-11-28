import numpy as np,pickle
from itertools import combinations,combinations_with_replacement
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; N,D=u.shape; y=p['v2']
lin=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def report(name,F,y=y):
 c=lin(F,y); R=y-F@c; h=np.sum(F*np.linalg.pinv(F).T,1); print(name, F.shape[1], np.around((R**2).mean(0)**.5,6), np.around(((R/np.maximum(1-h,1e-8)[:,None])**2).mean(0)**.5,6)); return c
O=np.c_[np.ones(N),u]; P=np.array([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(D),2)]).T; C=np.c_[O,P]; report('quad',C); print('coeff',lin(C,y).round(3))
for k in ['panel_a','panel_b']:
 a=s[k]['u']; b=s[k]['v']; f=np.c_[np.ones(len(a)),a]; print(k,report(k,f,b).round(4)); print('ranges',a.min(),a.max())
M=lin(np.c_[np.ones(len(s['panel_b']['u'])),s['panel_b']['u']],s['panel_b']['v'])[1:];
for name,w in [('orig',u),('b',u@M),('binv',u@np.linalg.pinv(M)),('bt',u@M.T),('v1',p['v1'])]:
 for fn,fun in [('rat2',lambda a:w*w/(1+a*w*w)),('sin',lambda a:np.sin(a*w)),('cos',lambda a:np.cos(a*w)),('ratodd',lambda a:w/(1+a*w*w)),('tanh',lambda a:np.tanh(a*w))]:
  for A in (O,C):
   def err(a):
    F=np.c_[A,fun(a)]; return np.linalg.norm(F@lin(F,y)-y)
   grid=np.linspace(.1,10,120); i=np.argmin([err(z) for z in grid]); opt=minimize_scalar(err,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); report((name,fn,round(opt.x,4)),np.c_[A,fun(opt.x)])
